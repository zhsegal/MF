# 20_skin/28_tme_degradome_mc.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/28_tme_degradome_mc.ipynb` and is never run._

# 28 — The cutaneous degradome in MF skin, on the cluster-level malignancy call

**Merged 2026-10-22** from `28_tme_degradome_mc` (build + tests), `28b_tme_degradome_figures_mc`
(figures) and `28c_amoeboid_motility_mc` (amoeboid motility). The pre-merge notebooks are in
`notebooks/old/*_premerge.ipynb`; nothing in the analysis changed, so re-running this notebook must
reproduce `deg28mc_*`, `deg28bmc_*` and `amb28cmc_*` byte-for-byte in their values.

| part | what | object | artifacts |
|---|---|---|---|
| **I** | build gate, arms, coverage, the tests | `data/degradome/skin_degradome.h5ad` | `tables/deg28mc_*` |
| **II** | figures + the healthy-CD4 arm | same object, Part I's tables | `figures/degradome/deg28bmc_*`, `tables/deg28bmc_*` |
| **III** | amoeboid motility machinery | `atlas_joint/skin_T_tcr_annotated_v4.h5ad` (backed) | `figures/amoeboid/amb28cmc_*`, `tables/amb28cmc_*` |

**Each part's §0 is self-contained** — it re-resolves `NB_DIR`, re-imports the helpers and re-loads
what it needs. To redraw a figure, run Part II §0 and then Part II only; you do not have to run
Part I. § numbering restarts inside each part, as in the pre-merge notebooks.

**Malignancy.** All three parts split CD4 on `mal_cluster` (26b), which became the project headline
on 2026-10-22 (`helpers/malignancy.py`). The degradome object on disk was *built* when the headline
was `mal_knn`, so Parts I and II still re-derive the CD4 split in memory — that re-split is now
idempotent with a fresh build, and it is kept so the notebook does not silently change meaning
depending on when the object was last rebuilt.

> **HEAVY** — Part I loads the degradome object and Part III reads a backed 42,347-gene T object.
> Run on a compute kernel, not the login node.

# Part I — build gate, arms, coverage, tests  (was `28_tme_degradome_mc`)

A fresh copy of `28_tme_degradome` with one change of input: the malignant-vs-reactive CD4 arms
come from **`mal_cluster`** (`26b_cluster_malignancy`) instead of `mal_tcr`. Everything else is
the same — panel, gates, tests, designs. 28 is left untouched, so the two can be compared later.
The outputs are `tables/deg28mc_*`.

`docs/cutaneous-matrix-summary.md` describes the cutaneous matrix as a set of compartments:
- fibrillar and basement-membrane collagens
- elastic fibre
- proteoglycan and matricellular scaffolds
- the enzyme classes that dismantle each of them
- hyaluronan turnover
- the adhesion machinery a lymphocyte uses to move through what is left

This notebook asks, gene by gene, how each compartment differs between the disease arm and healthy
skin, which signals the malignant clone sends into the stroma, and whether the stroma tracks how
much tumour a biopsy carries.

| § | comparison | design | replicate |
|---|---|---|---|
| **A** | CD4 malignant vs reactive | paired **within biopsy**; rerun on ALICE-decided clusters only | sample |
| **B** | CD4 malignant / reactive / external healthy; myeloid and fibroblast MF / HC / external healthy, at both resolutions | unpaired, per arm pair | donor |
| **C** | curated ligand–receptor edges, CD4 ↔ fibroblast and myeloid | paired within biopsy, plus a specificity null | sample |
| **D** *(new)* | tumour burden → stromal degradome families, balances and composition | Spearman across MF samples, pooled and within study | sample |

§0–§2 are gates and setup. §A–§D compute and write tables. **§R is the only place results appear.**

## The malignancy call — `mal_cluster`

Each sample's T cells are Leiden-clustered (res 1.0), and each cluster is called malignant or benign:
- in ALICE samples, by the ALICE family share of its TCR+ cells (`call_source = tcr`);
- elsewhere, by a gene program + V-gene skew learned on ALICE clusters, when its held-out score is
  clear (`auto`);
- otherwise, by a manual call (`user`).

HC is benign by definition. For the paired arms, 88 of the 100 MF samples carry both arms, against
33 on `mal_tcr`, and they come from 11 studies, including the 7 with no V(D)J.

- **Arms:** `malignant` vs `reactive` CD4, MF samples only. CD4 with `mal_cluster` undefined are
  excluded.
- **Circularity:** 11 panel genes are part of the program that calls `auto` clusters (TOX, CXCL13,
  CD70, COL6A3, ITGA4, GZMK, CCL4, CCL5, FOXP3, ITGA6, IL7R). A malignant-vs-reactive difference in
  those genes is partly built in. Tables carry `in_call_program`, and §A is rerun on
  `call_source == "tcr"` cells only, where the call is ALICE's.
- **The object on disk was built on `mal_knn`,** so the notebook derives its own arm column and writes
  it over `cfg.GROUPBY` in memory. The coverage gates therefore see the arms actually tested. Stromal
  levels are identical under any call.
- **No HC false-positive QC:** HC is benign by construction under `mal_cluster`. §2 reports
  concordance with `mal_tcr` on CDR3-bearing CD4 instead.

## Unit of analysis

Expression is `X` = log1p CP10K, with the size factor summed over all 42,347 genes **before** the
panel subset; §0 gates this.
- **§A and §C:** the replicate is the **sample**, with one `malignant − reactive` delta within each
  biopsy.
- **§B:** the replicate is the **donor**, and the test is not blocked by study.
- **§D:** the replicate is the **sample**. Burden is the malignant share of the sample's
  `mal_cluster`-defined T cells, all lineages.

Direction is carried by `pct_up`, never by a zero-inflated median. An abstention is a coverage
statement, not a null.

## Run order

§0 build gate → §1 grouping → §2 coverage, cohort, QC → §3 external reference → §A → §B → §C → §D →
§R. The build is `jobs/run_degradome_build.sh`, and the external object comes from
`jobs/run_external_degradome_build.sh`.

```python  # cell 2
# ============================================================================
# §0  Setup + the build gate
# ============================================================================
# assert_build_equivalence is THE gate: it proves the library size was summed over all 42,347
# genes before the columns were subset. If it fails, every number below is scaled by a
# cell-type-dependent factor and none of them mean anything.
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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

import degradome_data as cfg          # config
import degradome_helpers as D         # build + gate
import degradome_cells as DC          # the single-cell analysis layer
import malignancy as MAL              # the calls
import cluster_malig_helpers as CM    # mal_cluster provenance + the calling program

# Every table, figure and run log of this notebook goes to deg28mc_*, never over nb28's deg28_*.
cfg.TAB_PREFIX = cfg.FIG_PREFIX = "deg28mc_"
cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)

# The one call this analysis uses. cfg.MALIG_HEADLINE stays mal_knn because that is what the
# OBJECT was built with; the CD4 split is re-derived in memory (§1).
ARM_CALL = "mal_cluster"
MALIG, REACT = cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE
ARM_NAME = {MALIG: "malignant", REACT: "reactive"}
ARM_LEVEL = "arm_level"

adata = D.load_degradome_adata()
eq = D.cached("equivalence",
              lambda: {"max_abs_diff": float(D.assert_build_equivalence(adata))})
assert eq["max_abs_diff"] < 1e-6, f"normalization gate failed: {eq}"

PANEL = [g for g in cfg.panel_genes() if g in set(map(str, adata.var_names))]
CALL_PROGRAM = set(pd.read_csv(CM.PROGRAM_CSV)["gene"])
PANEL_IN_PROGRAM = sorted(set(PANEL) & CALL_PROGRAM)
print(f"\nnormalization gate ok: max |lognorm(panel) - lognorm(full)| = "
      f"{eq['max_abs_diff']:.3g}")
print(f"scope {cfg.DISEASES} skin | {adata.n_obs:,} cells, "
      f"{adata.obs[cfg.DONOR_KEY].nunique()} donors, "
      f"{adata.obs[cfg.SAMPLE_KEY].nunique()} samples, "
      f"{adata.obs[cfg.STUDY_KEY].nunique()} studies | panel {len(PANEL)} genes")
print(f"analysis call: {ARM_CALL} (the object was built on {cfg.MALIG_HEADLINE}; "
      f"only the CD4 rows differ) | panel genes in the calling program: {PANEL_IN_PROGRAM}")
```

```
nb22 sidecar reconciled: 20 levels, 328,461 cells, zero drift

skin_degradome.h5ad: 985,273 cells x 503 genes, 112 donors, 123 samples, 12 studies
  scope ['MF', 'HC']: kept 1,002,481 of 1,345,527 skin cells; dropped 17,208 with no identity
  DEGRADOME-ONLY object: var is a hand-curated 503-gene panel. Not for HVG selection, PCA, UMAP, clustering or genome-wide DE. Use skin_degradome_pseudobulk_full.parquet for anything needing all genes.
[cache] equivalence <- equivalence.json

normalization gate ok: max |lognorm(panel) - lognorm(full)| = 0
scope ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies | panel 503 genes
analysis call: mal_cluster (the object was built on mal_knn; only the CD4 rows differ) | panel genes in the calling program: ['CCL4', 'CCL5', 'CD70', 'COL6A3', 'CXCL13', 'FOXP3', 'GZMK', 'IL7R', 'ITGA4', 'ITGA6', 'TOX']
```

```python  # cell 3
# ============================================================================
# §1  The grouping: CD4 re-split on mal_cluster (the object stays as built)
# ============================================================================
# The vocabulary is unchanged (CD4_malignant / CD4_reactive / CD4_unassessed), so every gate and
# ccc_axis.CELL_MAP keep working; only the provenance of the CD4 split changes. The arm column is
# also written over cfg.GROUPBY in memory, so the coverage gates in §2 count the arms that are
# actually tested instead of the object's built-in mal_knn split.
CELL_ID = (adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs
           else pd.Series(adata.obs_names, index=adata.obs_names))
mal = MAL.load_malignancy(CELL_ID, has_tcr=adata.obs.get("has_tcr"), verbose=True)
mal.index = adata.obs_names

IS_CD4 = (adata.obs[cfg.CELLTYPE_SRC].astype(str) == "CD4").to_numpy()
BUILT_LEVEL = adata.obs[cfg.GROUPBY].astype(str).to_numpy()      # mal_knn split, QC only
LVL = pd.Series(BUILT_LEVEL, index=adata.obs_names)
LVL = LVL.mask(IS_CD4, pd.Series(MAL.cd4_level(adata.obs[cfg.CELLTYPE_SRC], mal,
                                               call=ARM_CALL).astype(str),
                                 index=adata.obs_names))
adata.obs[ARM_LEVEL] = pd.Categorical(LVL)
adata.obs[cfg.GROUPBY] = adata.obs[ARM_LEVEL]
LV = LVL.to_numpy()
IS_MF = (adata.obs["disease"].astype(str) == "MF").to_numpy()
IS_HC = (adata.obs["disease"].astype(str) == cfg.HC_DISEASE).to_numpy()

# which evidence decided each cell's cluster: tcr (ALICE) / auto (program + V skew) / user / hc
_src = pd.read_parquet(CM.CALLS_PQ, columns=["cell_id", "call_source"]).set_index("cell_id")
adata.obs["call_source"] = pd.Categorical(
    _src["call_source"].reindex(CELL_ID.to_numpy()).fillna("none").to_numpy())
SRC = adata.obs["call_source"].astype(str).to_numpy()

_arm = IS_MF & np.isin(LV, [MALIG, REACT])
print(f"\nMF CD4 in the arms: {int(_arm.sum()):,} | by call source:")
print(pd.crosstab(pd.Series(np.vectorize(ARM_NAME.get)(LV[_arm]), name="arm"),
                  pd.Series(SRC[_arm], name="call_source")).to_string())
```

```
has_tcr: 78,442 cells marked TCR-bearing by the V(D)J fold-in cache that atlas obs calls untested (li2024)

             n_defined  frac_defined  n_malignant  frac_malignant_where_defined
mal_tcr       142787.0        0.1449      68843.0                        0.4821
mal_cnv            0.0        0.0000          0.0                           NaN
mal_knn       326068.0        0.3309     215379.0                        0.6605
mal_cluster   405280.0        0.4113     198800.0                        0.4905


MF CD4 in the arms: 323,307 | by call source:

call_source   auto    tcr  user
arm                            
malignant    99101  83491  3051
reactive     42475  88254  6935
```

```python  # cell 4
# ============================================================================
# §2  Coverage gate, the cohort, malignancy QC, per-sample burden
# ============================================================================
# "no donor carries 25 cells of this level" and "the effect was null" are different findings.
# CLAIM is the set of levels anything below is allowed to make a statement about.
cov = D.coverage_table(adata.obs, out=cfg.tab("coverage"))
ab = D.abstentions(cov, out=cfg.tab("abstentions"))

CLAIM = [lv for lv in cfg.CLAIM_LEVELS
         if lv in set(cov.loc[cov["claimable"], cfg.GROUPBY])]
FIBRO = [lv for lv in cfg.FIBRO_LEVELS if lv in CLAIM]
MYELOID = [lv for lv in cfg.MYELOID_LEVELS if lv in CLAIM]
BELOW_GATE = sorted(set(cfg.CLAIM_LEVELS) - set(CLAIM))
print(f"claimable: {len(CLAIM)} of {len(cfg.CLAIM_LEVELS)} levels | "
      f"fibroblast {FIBRO} | myeloid {MYELOID}")
print(f"below the donor gate (reported, never quoted): {BELOW_GATE or 'none'}")

conc = MAL.concordance(mal, by=adata.obs[cfg.STAGE_KEY].astype(str).to_numpy())
conc.to_csv(cfg.tab("malignancy_concordance"), index=False)

# ---- QC only: mal_cluster against the ALICE sequence call on the same CDR3-bearing MF CD4.
# In ALICE samples mal_cluster IS the ALICE family share at cluster level, so disagreement here
# measures cluster impurity (a reactive cell inside a malignant cluster), not a second opinion.
_both = IS_MF & IS_CD4 & mal["mal_tcr_defined"].to_numpy() & mal["mal_cluster_defined"].to_numpy()
_t, _c = mal["mal_tcr"].to_numpy()[_both], mal["mal_cluster"].to_numpy()[_both]
TCR_AGREE = float((_t == _c).mean()) if _both.any() else np.nan
print(f"\nmal_cluster vs mal_tcr on {int(_both.sum()):,} CDR3-bearing MF CD4: agree {TCR_AGREE:.1%} | "
      f"ALICE-clonal called reactive {float((_t & ~_c).sum() / max(_t.sum(), 1)):.1%} | "
      f"ALICE-non-clonal called malignant {float((~_t & _c).sum() / max((~_t).sum(), 1)):.1%}")

_samp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
CALL_SAMPLES = sorted(set(_samp[IS_MF & IS_CD4 & mal["mal_cluster_defined"].to_numpy()]))
_cd4_mf = IS_MF & IS_CD4
CALL_COHORT = {
    "MF samples with a call": f"{len(CALL_SAMPLES)} of {len(set(_samp[IS_MF]))}",
    "donors": int(adata.obs.loc[_cd4_mf & np.isin(_samp, CALL_SAMPLES),
                                cfg.DONOR_KEY].nunique()),
    "studies": sorted(adata.obs.loc[IS_MF & np.isin(_samp, CALL_SAMPLES),
                                    cfg.STUDY_KEY].astype(str).unique()),
    "malignant cells": int(((LV == MALIG) & IS_MF).sum()),
    "reactive cells": int(((LV == REACT) & IS_MF).sum()),
    "excluded, no call": int(((LV == cfg.CD4_UNASSESSED) & IS_MF).sum()),
}
print("\ncohort (the cells sections A and C rest on):")
for k, v in CALL_COHORT.items():
    print(f"  {k}: {v}")

# ---- per-sample tumour burden, the §D predictor: malignant share of the sample's
# mal_cluster-defined T cells, all T lineages, taken from the call table itself.
_calls = pd.read_parquet(CM.CALLS_PQ, columns=["sample_id", "mal_cluster"]).dropna()
BURDEN = (_calls.assign(m=_calls["mal_cluster"].astype(float))
          .groupby("sample_id")["m"].agg(burden="mean", n_T="size"))
_meta = (adata.obs.loc[IS_MF, [cfg.SAMPLE_KEY, cfg.DONOR_KEY, cfg.STUDY_KEY, cfg.STAGE_KEY]]
         .astype(str).drop_duplicates(cfg.SAMPLE_KEY).set_index(cfg.SAMPLE_KEY))
BURDEN = _meta.join(BURDEN, how="inner")
BURDEN.to_csv(cfg.tab("burden"))
print(f"\nburden: {len(BURDEN)} MF samples | quantiles "
      f"{BURDEN['burden'].quantile([0, .25, .5, .75, 1]).round(2).to_dict()} -> {cfg.tab('burden').name}")

# how the cluster call and the object's built-in classifier split divide the same CD4 cells
print(f"\n{ARM_CALL} level (rows) against the object's {cfg.MALIG_HEADLINE} level (columns), MF CD4:")
print(pd.crosstab(pd.Series(LV[_cd4_mf], name=ARM_CALL),
                  pd.Series(BUILT_LEVEL[_cd4_mf], name=cfg.MALIG_HEADLINE)).to_string())
```

```
degradome_level  claim  n_cells  n_donors_HC  n_donors_early  n_donors_advanced  claimable  testable_HC_vs_advanced  testable_early_vs_advanced
             LC   True     3256            0               7                  9       True                    False                        True
           cDC1   True     7672            0              16                 22       True                    False                        True
           cDC2   True    14132           17              18                 29       True                     True                        True
       DC_LAMP3   True    22181            1              15                 23       True                    False                        True
            pDC   True      883            0               3                  6       True                    False                       False
      Mono_CD14   True     3596            0               7                 16       True                    False                        True
           moDC   True     6842            6               6                 24       True                     True                        True
      Mac_FOLR2   True    17584           15              18                 31       True                     True                        True
 Mac_SPP1_TREM2   True     3838            1               1                 13       True                    False                       False
       Mac_infl   True     8793            2               8                 13       True                    False                        True
    Neutrophils   True      293            0               0                  2      False                    False                       False
    F_papillary   True    12987           17              10                 15       True                     True                        True
    F_reticular   True    25136           17              13                 18       True                     True                        True
  F_mesenchymal   True     8259           18               8                 17       True                     True                        True
F_myofibroblast   True      597            0               0                  5       True                    False                       False
 F_inflammatory   True    45237           19              15                 21       True                     True                        True
        F_apCAF   True    17800           14              13                 23       True                     True                        True
  CD4_malignant   True   185643            0              19                 37       True                    False                        True
   CD4_reactive   True   141482           17              19                 38       True                     True                        True
 CD4_unassessed  False     1092            5               0                  1       True                    False                       False
       CD4_Treg  False    35086            3              16                 35       True                    False                        True
            CD8  False    43858            8              17                 37       True                     True                        True
              B  False    24810            0              11                 26       True                    False                        True
         Plasma  False    11481            1              15                 24       True                    False                        True
   Keratinocyte  False   212693           21              17                 31       True                     True                        True
       Vascular  False   114799           21              17                 37       True                     True                        True
           Mast  False     3483            6               5                  9       True                     T
... [3681 more chars truncated]
```

```python  # cell 5
# ============================================================================
# §3  External healthy reference + the arm masks every section below shares
# ============================================================================
# Reynolds 2021 healthy adult skin, same normalisation contract as ours. Its labels are coarser,
# so a level with no exact match is compared with its nearest parent, POOLED (cfg.EXT_POOLED);
# pDC and Neutrophils have no counterpart at all and get no external arm.
import external_ref as X

ext = X.load_external()
prov = X.provenance()
EXT_T = ext.obs[X.TARGET_COL].astype(str).to_numpy()
_ext_var = set(map(str, ext.var_names))
EXT_MISSING = [g for g in PANEL if g not in _ext_var]
ON_OBJ = set(LV)
print(f"{prov.get('paper', cfg.EXT_NAME)} | {ext.n_obs:,} cells | "
      f"{ext.obs[cfg.EXT_DONOR_KEY].nunique()} donors | {prov.get('assay', '?')} | "
      f"panel genes absent from it: {len(EXT_MISSING)}")


def tme_rows(groups):
    """{row: {MF, HC, external HC}} for a {row_label: [fine levels]} map, plus per-row notes."""
    rows, notes = {}, {}
    for row, levels in groups.items():
        levels = [lv for lv in levels if lv in ON_OBJ]
        if not levels:
            continue
        m = np.isin(LV, levels)
        tg = cfg.EXT_ARM.get(row)
        rows[row] = {"MF": (adata, m & IS_MF), "HC": (adata, m & IS_HC),
                     "external HC": (ext, np.isin(EXT_T, tg)) if tg else None}
        note = []
        if row in cfg.EXT_POOLED:
            note.append(f"ext pooled: {'+'.join(tg)}")
        if not set(levels) & set(CLAIM):
            note.append("below donor gate")
        if note:
            notes[row] = "; ".join(note)
    return rows, notes


CD4_ARMS = ["malignant", "reactive", "external HC"]
TME_ARMS = ["MF", "HC", "external HC"]

cd4_rows = {"CD4": {
    "malignant": (adata, (LV == MALIG) & IS_MF),
    "reactive": (adata, (LV == REACT) & IS_MF),
    "external HC": (ext, np.isin(EXT_T, cfg.EXT_ARM["CD4"])),
}}
mye_low, mye_low_notes = tme_rows(cfg.MYELOID_COARSE)
mye_high, mye_high_notes = tme_rows({lv: [lv] for lv in cfg.MYELOID_LEVELS})
fib_low, fib_low_notes = tme_rows(cfg.FIBRO_COARSE)
fib_high, fib_high_notes = tme_rows({lv: [lv] for lv in cfg.FIBRO_LEVELS})
print(f"blocks: cd4 (1 row) | myeloid_low ({len(mye_low)}) | myeloid_high ({len(mye_high)}) "
      f"| fibro_low ({len(fib_low)}) | fibro_high ({len(fib_high)})")
```

```
Reynolds et al., Science 2021;371:eaba6500 | 188,698 cells | 5 donors | 10x 3' v2 | panel genes absent from it: 2

blocks: cd4 (1 row) | myeloid_low (4) | myeloid_high (12) | fibro_low (3) | fibro_high (6)
```

```python  # cell 6
# ============================================================================
# §A  Intra-sample: CD4 malignant vs reactive, paired WITHIN each biopsy, per gene
# ============================================================================
# The replicate is the sample and the statistic is one malignant-minus-reactive difference inside
# each biopsy, so donor, study, chemistry, site and treatment are the same tube. A sample
# contributes only if BOTH arms clear cfg.MIN_CELLS.
#
# Run twice: on every called cell, and on cells whose cluster ALICE decided (call_source == tcr).
# The second run is the circularity check -- there no gene program took part in the call, so a
# gene in CALL_PROGRAM that holds up there is not an artefact of how the arms were drawn.
from statsmodels.stats.multitest import multipletests


def paired_panel(mask):
    """Per-gene paired malignant-vs-reactive test over the cells in `mask`; one BH over the panel."""
    sub = adata[mask].copy()
    parts = []
    for k in range(0, len(PANEL), 100):       # chunks of 100 keep the obs frame bounded
        cols = DC.gene_score_cells(sub, PANEL[k:k + 100], verbose=False)
        sm = DC.donor_median(sub, cols, groupby=ARM_LEVEL, donor_key=cfg.SAMPLE_KEY)
        parts.append(DC.paired_within(sm, cols, pair_key=cfg.SAMPLE_KEY, arm_col=ARM_LEVEL,
                                      ref=REACT, alt=MALIG, fdr=False))
        sub.obs.drop(columns=cols, inplace=True)
    t = pd.concat(parts, ignore_index=True)
    t["q"] = np.nan
    ok = t["p"].notna()
    if ok.any():
        t.loc[ok, "q"] = multipletests(t.loc[ok, "p"], method="fdr_bh")[1]
    t["gene"] = t["score"].str.replace("^gene_", "", regex=True)
    t["pct_up"] = t["n_pos"] / (t["n_pos"] + t["n_neg"]).replace(0, np.nan)
    # zero-inflated panels: direction is n_pos vs n_neg; an abstaining gene gets no direction
    t["higher"] = np.where(t["n_pos"] > t["n_neg"], "malignant", "reactive")
    t.loc[t["q"].isna(), "higher"] = ""
    return t, sub.obs[cfg.SAMPLE_KEY].nunique(), sub.obs[cfg.DONOR_KEY].nunique()


A_MASK = IS_MF & np.isin(LV, [MALIG, REACT])
A, _ns, _nd = paired_panel(A_MASK)
print(f"A cohort: {int(A_MASK.sum()):,} MF CD4 cells ({int(((LV == MALIG) & IS_MF).sum()):,} "
      f"malignant) in {_ns} samples, {_nd} donors")

A_TCR_MASK = A_MASK & (SRC == "tcr")
A_tcr, _ns_t, _ = paired_panel(A_TCR_MASK)
print(f"A, ALICE-decided clusters only: {int(A_TCR_MASK.sum()):,} cells in {_ns_t} samples")

A["family"] = A["gene"].map(cfg.GENE_TO_FAMILY).fillna("other")
A["in_call_program"] = A["gene"].isin(CALL_PROGRAM)
A = A.merge(A_tcr[["gene", "n_pairs", "pct_up", "q", "higher"]]
            .add_suffix("_tcr_src").rename(columns={"gene_tcr_src": "gene"}), on="gene", how="left")
A = A.sort_values("q")
A.to_csv(cfg.tab("A_paired_genes"), index=False)

A_NPAIRS = int(A["n_pairs"].max())
A_NPAIRS_TCR = int(A["n_pairs_tcr_src"].max())
_sig = A["q"] < 0.05
_rep = _sig & (A["q_tcr_src"] < 0.05) & (A["higher"] == A["higher_tcr_src"])
print(f"A: {A_NPAIRS} biopsies carry both arms above {cfg.MIN_CELLS} cells | "
      f"{int(_sig.sum())}/{int(A['p'].notna().sum())} genes at q<0.05, "
      f"{int(A['p'].isna().sum())} abstained | {int(_rep.sum())} of them also significant, same "
      f"direction, on ALICE-decided clusters ({A_NPAIRS_TCR} biopsies) -> {cfg.tab('A_paired_genes').name}")
```

```
A cohort: 323,307 MF CD4 cells (185,643 malignant) in 100 samples, 88 donors

A, ALICE-decided clusters only: 171,745 cells in 34 samples
A: 88 biopsies carry both arms above 25 cells | 84/126 genes at q<0.05, 377 abstained | 30 of them also significant, same direction, on ALICE-decided clusters (30 biopsies) -> deg28mc_A_paired_genes.csv
```

```python  # cell 7
# ============================================================================
# §B  Cross-arm, unpaired, per gene, at both resolutions
# ============================================================================
# The unit is the DONOR mean, never the cell: at 10^4 cells per arm a cell-level rank test calls
# everything significant and one deep patient can carry it. BH runs over the whole block
# (row x gene x arm pair). Not blocked by study -- MF vs HC mixes cohorts and the external arm is
# a different lab, chemistry and sex.
B_BLOCKS = {
    "cd4": (cd4_rows, CD4_ARMS, {"CD4": f"{ARM_CALL}, MF samples"}),
    "myeloid_low": (mye_low, TME_ARMS, mye_low_notes),
    "myeloid_high": (mye_high, TME_ARMS, mye_high_notes),
    "fibro_low": (fib_low, TME_ARMS, fib_low_notes),
    "fibro_high": (fib_high, TME_ARMS, fib_high_notes),
}

_parts, B_INFO, B_NOTES = [], {}, {}
for tag, (rows, arms, notes) in B_BLOCKS.items():
    t, info = DC.arm_test_block(rows, PANEL, arms, label=tag)
    t.insert(0, "block", tag)
    _parts.append(t)
    B_INFO[tag] = info
    B_NOTES[tag] = notes

B = pd.concat(_parts, ignore_index=True)
B["family"] = B["gene"].map(cfg.GENE_TO_FAMILY).fillna("other")
B["in_call_program"] = (B["block"] == "cd4") & B["gene"].isin(CALL_PROGRAM)
B["arm_pair"] = B["arm_a"] + " vs " + B["arm_b"]
B = B.sort_values(["block", "arm_pair", "q"])
B.to_csv(cfg.tab("B_arm_tests"), index=False)
print(f"\nB: {int((B.status == 'significant').sum())}/{int((B.status != 'not_tested').sum())} "
      f"tested rows at q<0.05, {int((B.status == 'not_tested').sum())} not tested "
      f"-> {cfg.tab('B_arm_tests').name}")
```

```
cd4: 299/503 genes above 1% detection; 226/897 donor-level tests at q < 0.05; never detected: 204; absent from external var: 2

myeloid_low: 410/503 genes above 1% detection; 1026/2861 donor-level tests at q < 0.05; never detected: 93; absent from external var: 2

myeloid_high: 459/503 genes above 1% detection; 1762/6834 donor-level tests at q < 0.05; never detected: 44; absent from external var: 2

fibro_low: 420/503 genes above 1% detection; 1760/3778 donor-level tests at q < 0.05; never detected: 83; absent from external var: 2

fibro_high: 441/503 genes above 1% detection; 2894/7041 donor-level tests at q < 0.05; never detected: 62; absent from external var: 2


B: 7668/21411 tested rows at q<0.05, 12648 not tested -> deg28mc_B_arm_tests.csv
```

```python  # cell 8
# ============================================================================
# §C  Ligand-receptor traffic, CD4 <-> fibroblast and myeloid only
# ============================================================================
# Expression comes from the full-gene (level x sample) cube the degradome build writes in the
# same pass as the object -- same cells, same scope, one vintage -- EXCEPT the two CD4 arm rows,
# which the cube split on mal_knn and which cube_tcr re-derives from arm_level. A stromal level
# is recomputed from the object and asserted equal to the cube's own row, or the two halves are
# on different scales and every delta below is a units artefact.
#
# The score is sqrt(ligand * receptor) per biopsy, a complex taking the min over its subunits,
# and the test is the section A design. There is NO specificity rank over a resource here -- 30
# curated edges cannot supply one -- but there is a null drawn from the transcriptome: see
# edge_specificity.
import pyarrow.parquet as pq

import ccc_axis as CA

STROMA_LEVELS = list(cfg.FIBRO_LEVELS) + list(cfg.MYELOID_LEVELS)
_cube_cols = [c for c in pq.ParquetFile(cfg.PSEUDOBULK_PQ).schema_arrow.names
              if not c.startswith("__index_level")]
# A gene has to exist on BOTH halves of the cube to be scorable: the CD4 rows only carry the
# panel. Curated LR symbols are all on the panel by construction (cfg.panel_genes includes them).
_scorable = sorted(set(_cube_cols) & set(map(str, adata.var_names)))

lr_pairs, lr_dropped = CA.priority_pairs_frame(var_names=_scorable,
                                               stroma_levels=STROMA_LEVELS)
lr_pairs.drop(columns=["ligand_genes", "receptor_genes"]).to_csv(cfg.tab("ccc_pairs"),
                                                                 index=False)
lr_dropped.to_csv(cfg.tab("ccc_pairs_dropped"), index=False)

_need = sorted({g for gs in list(lr_pairs.ligand_genes) + list(lr_pairs.receptor_genes)
                for g in gs})
lr_expr, lr_meta, lr_pools = CA.cube_tcr(adata, LV, genes=_need)
lr_scores = CA.edge_scores(lr_expr, lr_meta, lr_pairs)
C = CA.edge_paired_test(lr_scores)
C = CA.edge_specificity(lr_expr, lr_meta, lr_pairs, C, lr_pools)
C["pct_up"] = C["n_pos"] / (C["n_pos"] + C["n_neg"]).replace(0, np.nan)
C["higher"] = C["higher"].map(lambda v: ARM_NAME.get(v, v))
C = C.sort_values(["spec_p", "q"])
C.to_csv(cfg.tab("C_ccc_edges"), index=False)

C_SIG = C[C["q"] < 0.05]
C_SPEC = C_SIG[C_SIG["spec_p"] < 0.05]
print(f"\nC: {len(C_SIG)}/{int(C['q'].notna().sum())} edge x level rows at q<0.05, "
      f"of which {len(C_SPEC)} also beat both specificity nulls; "
      f"{int(C['q'].isna().sum())} abstained -> {cfg.tab('C_ccc_edges').name}")
```

```
priority pairs: 30 curated -> 62 concrete edges over 18 stromal levels (472 edge x level rows)
  not scored (4):
    [2] ligand: ADAM17->sIL6R -- trans-signalling by a shed receptor, not a transcript pair
    [9] receptor: TCR -- a rearranged receptor, not a gene symbol
    [14] cell: Keratinocyte -- stromal level outside the requested vocabulary
    [14] both: LAMA3->ITGA3.ITGB1 -- absent from the object: ['ITGA3']

cube: 1,400 pseudo-samples x 2,503 genes (size factor over all 42,347) | disease=MF | >= 25 cells | 103 samples, 90 donors, 11 studies

library size recovered for 985,273/985,273 cells (median 2,626)

scale gate on Keratinocyte: max |object - cube| = 8.84e-09 over 87 pseudo-samples x 503 genes

CD4 arms re-derived from the supplied level: CD4_malignant 90 samples, CD4_reactive 98 samples | null pools: CD4 side 503 panel genes, stromal side 2000 cube genes

  dropped 78 edge x level keys below 5 paired samples (levels touched: ['F_myofibroblast', 'Mono_CD16', 'Neutrophils'])
scored 35,166 (sample x arm x edge x level) rows | 394 edge x level keys | 83 samples carry both CD4 arms

paired edge tests: 394 edge x level, 218 at q < 0.05 | median n_pairs = 44

specificity: 9/394 edge x level rows beat both nulls at p < 0.05 (CD4-side pool 503 panel genes -- conservative; stromal-side pool 2000 cube genes)

C: 218/393 edge x level rows at q<0.05, of which 8 also beat both specificity nulls; 1 abstained -> deg28mc_C_ccc_edges.csv
```

```python  # cell 9
# ============================================================================
# §D  Tumour burden -> stroma, across MF samples
# ============================================================================
# mal_cluster calls every MF sample, so burden -- the malignant share of a sample's T cells --
# spans ~100 samples from 0 to 1. If the clone instructs the stroma, stromal remodelling should
# track it. Two readouts per stromal row (fibroblast / myeloid, coarse resolution):
#   state        per-sample median of each degradome family score and protease/inhibitor balance
#   composition  the share of its compartment the row makes up in that sample
# Spearman across samples, pooled and within study (ranks centred inside each study with >= 4
# samples, so a between-study shift in both burden and stroma cannot make the correlation).
# BH over every row x readout, separately for the two versions. One sample = one replicate.
from scipy.stats import spearmanr

STROMA_COARSE = {**cfg.FIBRO_COARSE, **cfg.MYELOID_COARSE}
COMPARTMENT = {**{r: "fibroblast" for r in cfg.FIBRO_COARSE},
               **{r: "myeloid" for r in cfg.MYELOID_COARSE}}
_to_row = {lv: row for row, lvs in STROMA_COARSE.items() for lv in lvs}
D_MASK = IS_MF & np.isin(LV, list(_to_row)) & np.isin(_samp, BURDEN.index)
st = adata[D_MASK].copy()
st.obs["stroma_row"] = st.obs[ARM_LEVEL].astype(str).map(_to_row)

fams = DC.family_score_cells(st, families=list(cfg.DEGRADOME_FAMILIES), verbose=False)
bals = DC.balance_cells(st, verbose=False)
state_cols = [f"fam_{k}" for k in fams] + [f"bal_{k}" for k in bals]
smed = DC.donor_median(st, state_cols, groupby="stroma_row", donor_key=cfg.SAMPLE_KEY)
state_long = (smed[smed["eligible"]]
              .melt(id_vars=["stroma_row", cfg.SAMPLE_KEY], value_vars=state_cols,
                    var_name="readout", value_name="value"))

_cnt = pd.crosstab(st.obs[cfg.SAMPLE_KEY].astype(str), st.obs["stroma_row"])
_comp = []
for cp in ("fibroblast", "myeloid"):
    rows_ = [r for r in _cnt.columns if COMPARTMENT.get(r) == cp]
    tot = _cnt[rows_].sum(axis=1)
    keep = tot >= cfg.MIN_CELLS                       # a share of < 25 compartment cells is noise
    sh = _cnt.loc[keep, rows_].div(tot[keep], axis=0)
    _comp.append(sh.stack().rename("value").reset_index().assign(readout=f"share_of_{cp}"))
comp_long = pd.concat(_comp, ignore_index=True)

long = (pd.concat([state_long, comp_long], ignore_index=True)
        .merge(BURDEN[["burden", cfg.STUDY_KEY]], left_on=cfg.SAMPLE_KEY, right_index=True))


def _within_rank(v, g):
    """Percentile ranks centred within each study; studies with < 4 samples drop out."""
    r = v.groupby(g).rank(pct=True)
    return (r - r.groupby(g).transform("mean")).where(g.map(g.value_counts()) >= 4)


_rows = []
for (row, readout), d in long.groupby(["stroma_row", "readout"]):
    d = d.dropna(subset=["value"])
    rec = {"stroma_row": row, "compartment": COMPARTMENT[row], "readout": readout,
           "n_samples": len(d), "n_studies": d[cfg.STUDY_KEY].nunique()}
    if len(d) < 10 or d["value"].nunique() < 3:
        _rows.append({**rec, "status": "not_tested"})
        continue
    rec["rho"], rec["p"] = spearmanr(d["burden"], d["value"])
    wb, wv = _within_rank(d["burden"], d[cfg.STUDY_KEY]), _within_rank(d["value"], d[cfg.STUDY_KEY])
    ok = wb.notna() & wv.notna()
    rec["n_within"] = int(ok.sum())
    if ok.sum() >= 10:
        rec["rho_within_study"], rec["p_within_study"] = spearmanr(wb[ok], wv[ok])
    _rows.append({**rec, "status": "tested"})

Dt = pd.DataFrame(_rows)
Dt = Dt.reindex(columns=list(dict.fromkeys(
    [*Dt.columns, "rho", "p", "n_within", "rho_within_study", "p_within_study"])))
for pc, qc in (("p", "q"), ("p_within_study", "q_within_study")):
    Dt[qc] = np.nan
    m = Dt[pc].notna()
    if m.any():
        Dt.loc[m, qc] = multipletests(Dt.loc[m, pc].astype(float), method="fdr_bh")[1]
# a row counts only if it holds pooled AND within study, in the same direction
Dt["robust"] = ((Dt["q"] < 0.05) & (Dt["q_within_study"] < 0.05)
                & (np.sign(Dt["rho"]) == np.sign(Dt["rho_within_study"])))
Dt = Dt.sort_values(["robust", "q"], ascending=[False, True])
Dt.to_csv(cfg.tab("D_burden_stroma"), index=False)
D_SIG, D_ROBUST = Dt[Dt["q"] < 0.05], Dt[Dt["robust"]]
print(f"\nD: {int((Dt.status == 'tested').sum())} row x readout tests over "
      f"{long[cfg.SAMPLE_KEY].nunique()} MF samples | {len(D_SIG)} at q<0.05 pooled, "
      f"{len(D_ROBUST)} also within study -> {cfg.tab('D_burden_stroma').name}")
del st
```

```
D: 131 row x readout tests over 98 MF samples | 0 at q<0.05 pooled, 0 also within study -> deg28mc_D_burden_stroma.csv
```

```python  # cell 10
# ============================================================================
# §R  RESULTS — the only cell that reports anything
# ============================================================================
# Written to be read on its own, by a reader (or an agent) with no access to the notebook: the
# preamble states the question, the data, the panel, the arms, the designs and every column,
# then the comparisons that were run, then the significant rows. A non-significant number
# appears nowhere except as a count in the inventory.
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 4000)
L = []
P = L.append
RULE = "=" * 100

_fam_present = {}
_on_obj = set(map(str, adata.var_names))
for _name, _keys in cfg.FAMILY_ORDER.items():
    _gs = [g for k in _keys for g in cfg.DEGRADOME_FAMILIES[k] if g in _on_obj]
    _fam_present[_name] = _gs

P("#" * 100)
P("NB28_mc — THE CUTANEOUS DEGRADOME IN MYCOSIS FUNGOIDES SKIN, ON THE CLUSTER-LEVEL MALIGNANCY CALL")
P("SELF-CONTAINED RESULTS REPORT — everything needed to interpret these numbers is below.")
P("#" * 100)

# ---------------------------------------------------------------- 1. the question
P("")
P(RULE)
P("1 · THE QUESTION")
P(RULE)
P("Mycosis fungoides (MF) is a cutaneous T-cell lymphoma in which a malignant CD4 clone sits in")
P("the skin alongside reactive T cells, myeloid cells and fibroblasts. The extracellular matrix")
P("around it is remodelled. This notebook asks four things over a curated 'degradome' panel --")
P("the proteases that dismantle each matrix compartment, their inhibitors, the matrix substrates,")
P("and the adhesion/motility machinery a lymphocyte uses to move through what is left:")
P("  (A) inside one biopsy, what does the malignant clone carry that its reactive neighbours do")
P("      not?")
P("  (B) across donors, how do MF cells of each type differ from healthy-control cells of the")
P("      same type, in this atlas and in an independent healthy skin dataset?")
P("  (C) which ligand-receptor edges between the clone and the stroma strengthen in the clone,")
P("      specifically rather than as a by-product of abundance?")
P("  (D) across MF samples, does the stroma track how much tumour the biopsy carries?")

# ---------------------------------------------------------------- 2. the data
P("")
P(RULE)
P("2 · THE DATA")
P(RULE)
P(f"Atlas v2, skin only, disease in {cfg.DISEASES} (MF patients and healthy controls) -- the scope")
P("the degradome object was built on. SS and other CTCL entities are not in this object.")
P(f"  {adata.n_obs:,} cells | {adata.obs[cfg.DONOR_KEY].nunique()} donors | "
  f"{adata.obs[cfg.SAMPLE_KEY].nunique()} samples | "
  f"{adata.obs[cfg.STUDY_KEY].nunique()} studies")
P(f"EXPRESSION UNITS: log1p(CP10K). The library size is summed over all 42,347 genes BEFORE the")
P(f"  panel columns are subset, so values are comparable across cell types and sequencing")
P(f"  depths. This is gated, not assumed: max |lognorm(panel) - lognorm(full)| = "
  f"{eq['max_abs_diff']:.3g}.")
P(f"EXTERNAL HEALTHY REFERENCE: {cfg.EXT_NAME} "
  f"({prov.get('accession', cfg.EXT_ACCESSION)}), {ext.n_obs:,} healthy adult skin cells from")
P(f"  {ext.obs[cfg.EXT_DONOR_KEY].nunique()} donors, a different lab, chemistry and sex mix,")
P(f"  normalised on the same contract. Its cell labels are coarser than ours, so a level with")
P(f"  no exact match is compared with its nearest parent POOLED -- that is a reference level,")
P(f"  not a matched subset. Pooled levels: {sorted(cfg.EXT_POOLED)}.")
P(f"  No external counterpart at all: pDC, Neutrophils.")
P(f"  Panel genes absent from the reference ({len(EXT_MISSING)}): {EXT_MISSING or 'none'}.")

# ---------------------------------------------------------------- 3. the panel
P("")
P(RULE)
P("3 · THE GENE PANEL")
P(RULE)
P(f"{len(PANEL)} genes on the object. Sections A and B test EVERY one of them. Families, in")
P("display order, with the genes present here:")
for _name, _gs in _fam_present.items():
    P(f"  {_name:20s} ({len(_gs):2d})  {', '.join(_gs)}")
P("")
P("Panel extension (docs/NB28_panel_extension_spec.md): the signals the clone emits, the")
P("receptors the stroma answers with, and metabolic / transcription-factor controls. Same")
P("gates, same tests, same single BH family as the genes above:")
for _name, _gs in cfg.PANEL_EXTENSION.items():
    _gs = [g for g in _gs if g in _on_obj]
    if _gs:
        P(f"  {_name:20s} ({len(_gs):2d})  {', '.join(_gs)}")
P("")
P("Eight mechanistic fibrosis axes overlap the families and are carried on the same object.")
P("Each is split into a side a CD4 cell can EMIT and a side a fibroblast or myeloid cell")
P("ANSWERS with. A ligand is evidence only in a CD4 row and a receptor only in a stromal row:")
for _ax in cfg.FIBROSIS_ORDER:
    _sd = cfg.FIBROSIS_SIDE[_ax]
    P(f"  {_ax:24s} CD4 side: {'+'.join(_sd['ligand'])}")
    P(f"  {'':24s} stromal side: {'+'.join(_sd['stroma'])}")

# ---------------------------------------------------------------- 4. arms
P("")
P(RULE)
P("4 · THE MALIGNANCY CALL AND THE ARMS")
P(RULE)
P(f"This analysis uses ONE call: {ARM_CALL}. Every sample's T cells are Leiden-clustered on their")
P("own (res 1.0) and each CLUSTER is called malignant or benign; every cell takes its cluster's")
P("call. The evidence that decided a cluster is recorded per cell (call_source):")
P("  tcr   ALICE TCR clonality -- the share of the cluster's TCR+ cells in the donor's founder")
P("        family (>= 0.8 malignant, <= 0.1 benign). A sequence fact.")
P("  auto  a gene program + V-gene skew learned on the ALICE clusters, held-out AUC 0.99, used")
P("        only where its score was clear at <= 2% held-out error.")
P("  user  a manual call on a review sheet.")
P("  malignant  = mal_cluster True, CD4, MF samples")
P("  reactive   = mal_cluster False, CD4, MF samples")
P("  EXCLUDED   = CD4 with no call (tiny samples, entities out of scope, user-excluded samples)")
P("CIRCULARITY. The auto program includes panel genes; for those genes a malignant-vs-reactive")
P("difference is partly how the arms were drawn. They are flagged in_call_program, and section A")
P("is rerun on call_source == tcr cells only (*_tcr_src columns), where no program took part.")
P(f"  panel genes in the program: {PANEL_IN_PROGRAM}")
P(f"QC: on CDR3-bearing MF CD4 the cluster call agrees with the per-cell ALICE call on "
  f"{TCR_AGREE:.1%} of cells (disagreement = cluster impurity). HC is benign by construction, so")
P("there is no healthy-skin false-positive rate to report for this call.")
P("Cohort that sections A and C rest on: "
  + "; ".join(f"{k} {v}" for k, v in CALL_COHORT.items()))

# ---------------------------------------------------------------- 5. designs
P("")
P(RULE)
P("5 · THE FOUR COMPARISONS AND HOW EACH WAS TESTED")
P(RULE)
P("A · INTRA-SAMPLE, PAIRED. Replicate = the BIOPSY. One malignant-minus-reactive difference of")
P("    per-sample median expression, computed WITHIN each biopsy, then a two-sided Wilcoxon")
P("    signed-rank over those differences with a sign test alongside. Both arms come from the")
P("    same tube, so donor, study, chemistry, site, treatment and batch are fixed BY")
P(f"    CONSTRUCTION. A sample contributes only if both arms clear {cfg.MIN_CELLS} cells. BH over")
P(f"    all {len(PANEL)} genes. Repeated on ALICE-decided clusters only, with its own BH.")
P("B · CROSS-ARM, UNPAIRED. Replicate = the DONOR. Each donor's mean expression in an arm is one")
P("    observation; two-sided Mann-Whitney U between arms. There is deliberately NO cell-level")
P(f"    p-value anywhere. Each arm needs at least {cfg.MIN_DONORS_PER_ARM} donors with >= "
  f"{cfg.MIN_CELLS} cells.")
P("    BH over the whole block (every cell type x gene x arm pair in it). Two resolutions:")
P(f"      cd4          rows: CD4                      arms: {' / '.join(CD4_ARMS)}")
P(f"      myeloid_low  rows: {', '.join(mye_low)}")
P(f"      myeloid_high rows: {', '.join(mye_high)}")
P(f"      fibro_low    rows: {', '.join(fib_low)}")
P(f"      fibro_high   rows: {', '.join(fib_high)}")
P(f"      the four stromal blocks use arms: {' / '.join(TME_ARMS)}, where MF = cells of that")
P("      type from MF-patient biopsies and HC = cells of that type from healthy-control")
P("      biopsies in this same atlas.")
P("C · LIGAND-RECEPTOR, PAIRED, WITH A SPECIFICITY NULL. Replicate = the BIOPSY, same design as")
P("    A. Expression is pseudobulk per (cell type x biopsy). For each curated ligand-receptor")
P("    pair the score is sqrt(ligand_in_sender x receptor_in_receiver) within one biopsy, a")
P("    multi-subunit complex taking the minimum over its subunits; the contrast substitutes the")
P("    malignant and the reactive CD4 arm on the CD4 side of the edge.")
P(f"    Panel: {len(CA.panel.PRIORITY_PAIRS)} hand-curated edges -> {lr_pairs.edge.nunique()} "
  f"concrete edges x {lr_pairs.stroma_level.nunique()} stromal levels. Restricted to CD4 <-> ")
P("    fibroblast and CD4 <-> myeloid in both directions; nothing else is scored.")
P(f"    An edge is kept only if at least {cfg.MIN_SAMPLES} biopsies carry both CD4 arms for it.")
P("D · BURDEN -> STROMA, ACROSS SAMPLES. Replicate = the MF SAMPLE. Burden = malignant share of")
P("    the sample's mal_cluster-defined T cells (all lineages). Readouts per coarse stromal row:")
P("    the per-sample median of every degradome family score and protease/inhibitor balance")
P(f"    (row needs >= {cfg.MIN_CELLS} cells in the sample), and the row's share of its compartment.")
P("    Spearman across samples, pooled, and again on ranks centred within each study (>= 4")
P("    samples) so that a study that is both tumour-heavy and stroma-shifted cannot make the")
P("    correlation. BH over all row x readout tests, separately per version. 'robust' = q<0.05")
P("    in both, same sign.")

# ---------------------------------------------------------------- 6. how to read
P("")
P(RULE)
P("6 · HOW TO READ THESE NUMBERS — six rules")
P(RULE)
P("(1) DIRECTION IS pct_up, NOT THE MEDIAN. These panels are zero-inflated: median_delta is often")
P("    exactly 0 even where the signed-rank is decisive. pct_up is the share of NONZERO paired")
P("    differences that went up, and 'higher' names the arm.")
P("(2) AN ABSTENTION IS NOT A NULL. Untested rows are listed separately at the bottom.")
P("(3) SECTION B IS NOT BLOCKED BY STUDY. A significant B row means the donor distributions")
P("    separate, NOT that cohort and platform effects have been ruled out. A and C are immune to")
P("    this by construction; D's within-study version addresses it for D.")
P("(4) A HIGH LR SCORE IS NOT SPECIFICITY. Section C's two nulls separate a specific edge from an")
P("    abundant one; these numbers are NOT comparable to LIANA output from other notebooks.")
P("(5) ONLY SIGNIFICANT ROWS (BH q < 0.05) ARE PRINTED BELOW; the inventory counts the rest.")
P("(6) A GENE MARKED in_call_program IS PARTLY CIRCULAR in A and in the cd4 block of B. Read its")
P("    A result from the *_tcr_src columns (ALICE-decided clusters only), not the pooled ones.")
P("    D is a correlation across samples: it cannot say which side drives the other.")

# ---------------------------------------------------------------- 7. column glossary
P("")
P(RULE)
P("7 · COLUMN GLOSSARY")
P(RULE)
P("A (paired, per gene)")
P("  delta        median over biopsies of (malignant - reactive) per-sample median expression,")
P("               log1p CP10K. Often exactly 0 -- see rule (1).")
P("  pct_up       share of nonzero paired differences that were positive. THE direction column.")
P("  n_pairs      biopsies carrying both arms above the cell gate. n_pos / n_neg split them.")
P("  q            BH-adjusted Wilcoxon signed-rank p over all panel genes. sign_p is the sign")
P("               test on the same differences.")
P("  higher       'malignant' or 'reactive'; blank where the gene abstained.")
P("  in_call_program   the gene is part of the program that calls auto clusters (rule 6).")
P("  *_tcr_src    the same test on cells whose cluster ALICE decided, with its own BH.")
P("B (unpaired, per gene, per arm pair)")
P("  donor_median_a / _b   median across donors of each donor's mean expression, in arm_a and")
P("               arm_b respectively, log1p CP10K.")
P("  delta        donor_median_b - donor_median_a; read it together with 'higher'.")
P("  n_donors_a / _b       donors that actually entered the test in each arm -- the real n.")
P("  q            BH-adjusted Mann-Whitney p over the whole block.")
P("C (ligand-receptor, paired, with nulls)")
P("  direction    cd4_to_stroma = the clone is the SENDER. stroma_to_cd4 = the clone is the")
P("               RECEIVER; there the stromal ligand is identical between arms (same biopsy), so")
P("               the difference is driven entirely by the CD4 receptor.")
P("  edge         LIGAND->RECEPTOR. stroma_level is the non-CD4 partner cell type.")
P("  axis         which of the eight fibrosis axes in section 3 the edge speaks to.")
P("  delta/pct_up/n_pairs/q   as in A, on the edge score instead of a gene.")
P("  spec_p_cd4_side      CD4-side gene replaced by panel genes of the same abundance (conservative).")
P("  spec_p_stroma_side   stromal-side gene replaced by transcriptome genes of the same abundance.")
P("  spec_p       the LARGER of the two: an edge must beat BOTH nulls.")
P("D (burden -> stroma, per stromal row x readout)")
P("  readout      fam_<family> / bal_<balance> = per-sample median state score of that row;")
P("               share_of_<compartment> = the row's share of fibroblasts or myeloid cells.")
P("  rho, q       Spearman burden vs readout across samples, BH-adjusted.")
P("  rho_within_study, q_within_study   the same on within-study centred ranks.")
P("  n_samples / n_within / n_studies   samples entering each version, studies represented.")

# ---------------------------------------------------------------- 8. inventory
def _b_row(tag, label, arms):
    d = B[B.block == tag]
    t = d[d.status != "not_tested"]
    rep = (f"d={int(t[['n_donors_a', 'n_donors_b']].min().min())}-"
           f"{int(t[['n_donors_a', 'n_donors_b']].max().max())}" if len(t) else "-")
    return {"§": "B", "comparison": label, "arms": " / ".join(arms), "replicates": rep,
            "tested": len(t), "significant": int((d.status == "significant").sum()),
            "not tested": int((d.status == "not_tested").sum())}


inv = [{"§": "A", "comparison": "CD4, within biopsy", "arms": "malignant - reactive",
        "replicates": f"n={A_NPAIRS}", "tested": int(A["p"].notna().sum()),
        "significant": int((A["q"] < 0.05).sum()),
        "not tested": int(A["p"].isna().sum())},
       {"§": "A", "comparison": "CD4, within biopsy, ALICE-decided clusters",
        "arms": "malignant - reactive", "replicates": f"n={A_NPAIRS_TCR}",
        "tested": int(A["q_tcr_src"].notna().sum()),
        "significant": int((A["q_tcr_src"] < 0.05).sum()),
        "not tested": int(A["q_tcr_src"].isna().sum())}]
inv += [_b_row("cd4", "CD4", CD4_ARMS),
        _b_row("myeloid_low", f"myeloid low res ({len(mye_low)} types)", TME_ARMS),
        _b_row("myeloid_high", f"myeloid high res ({len(mye_high)} types)", TME_ARMS),
        _b_row("fibro_low", f"fibroblast low res ({len(fib_low)} types)", TME_ARMS),
        _b_row("fibro_high", f"fibroblast high res ({len(fib_high)} types)", TME_ARMS)]
inv += [{"§": "C", "comparison": "CD4 <-> fibro/myeloid LR edges",
         "arms": "malignant - reactive", "replicates": f"n<={int(C['n_pairs'].max())}",
         "tested": int(C["q"].notna().sum()), "significant": len(C_SIG),
         "not tested": int(C["q"].isna().sum())},
        {"§": "D", "comparison": "burden -> stromal state / composition",
         "arms": "Spearman across samples", "replicates": f"n<={int(Dt['n_samples'].max())}",
         "tested": int((Dt.status == "tested").sum()), "significant": len(D_ROBUST),
         "not tested": int((Dt.status == "not_tested").sum())}]
P("")
P(RULE)
P("8 · EVERY COMPARISON THAT WAS RUN — including the ones that produced nothing")
P(RULE)
P("'replicates' is n biopsies for A and C, the range of donors per arm for B, n samples for D.")
P("'significant' for D counts robust rows (pooled and within study).")
P(pd.DataFrame(inv).to_string(index=False))

# ---------------------------------------------------------------- A
_a = A[A["q"] < 0.05].sort_values("q")
P("")
P(RULE)
P(f"A · CD4 MALIGNANT vs REACTIVE, PAIRED WITHIN BIOPSY — {len(_a)} of "
  f"{int(A['p'].notna().sum())} genes at q<0.05 (n_pairs <= {A_NPAIRS})")
P(RULE)
P("Both arms are the same biopsy. q_tcr_src / higher_tcr_src repeat the test on ALICE-decided")
P(f"clusters only (n_pairs <= {A_NPAIRS_TCR}); prog = in_call_program (rule 6).")
if len(_a):
    P(_a[["gene", "family", "median_delta", "pct_up", "n_pairs", "q", "sign_p", "higher",
          "in_call_program", "q_tcr_src", "higher_tcr_src"]]
      .rename(columns={"median_delta": "delta", "in_call_program": "prog"})
      .round(4).to_string(index=False))
    P("")
    P("  by family: " + "; ".join(f"{k} {v}" for k, v in
                                  _a.groupby("family").size().sort_values(ascending=False)
                                  .items()))
    P("  higher in: " + "; ".join(f"{k} {v}" for k, v in _a["higher"].value_counts().items()))
    _repl = _a[(_a["q_tcr_src"] < 0.05) & (_a["higher"] == _a["higher_tcr_src"])]
    P(f"  replicated on ALICE-decided clusters (q<0.05, same direction): {len(_repl)} of {len(_a)}")
    _circ = _a[_a["in_call_program"]]
    if len(_circ):
        P("  circular genes and whether they hold without the program: "
          + "; ".join(f"{r.gene} {'holds' if (r.q_tcr_src < 0.05 and r.higher == r.higher_tcr_src) else 'does NOT hold'}"
                      for r in _circ.itertuples()))
else:
    P("  none.")

# ---------------------------------------------------------------- B
P("")
P(RULE)
P(f"B · PER-ARM COMPARISONS ACROSS DONORS — "
  f"{int((B.status == 'significant').sum())} of {int((B.status != 'not_tested').sum())} "
  f"rows at q<0.05")
P(RULE)
P("One sub-table per block and per arm pair. Row = cell type. Remember rule (3): these are not")
P("blocked by study. Rows flagged 'below donor gate' are drawn from too few donors to quote;")
P("rows flagged 'ext pooled' are compared against a pooled parent population.")
for tag in B_BLOCKS:
    d = B[(B.block == tag) & (B.status == "significant")]
    note = B_NOTES.get(tag) or {}
    P("")
    P(f"--- {tag} | {len(d)} significant" + (f" | notes: {note}" if note else ""))
    if not len(d):
        P("    none.")
        continue
    for pair, dp in d.groupby("arm_pair", sort=False):
        P(f"  {pair}  ({len(dp)})")
        P(dp.sort_values("q")[["row", "gene", "family", "donor_median_a", "donor_median_b",
                               "delta", "n_donors_a", "n_donors_b", "q", "higher",
                               "in_call_program"]]
          .rename(columns={"in_call_program": "prog"}).round(4).to_string(index=False))

_pc = B[(B.block == "myeloid_high") & (B.row == "Neutrophils")
        & (B.gene.isin(["ELANE", "PRTN3", "CTSG"]))]
P("")
P("  POSITIVE CONTROL — the neutrophil granule proteases ELANE / PRTN3 / CTSG in the")
P("  Neutrophils row. These must be carried by that population; if they are not, the cell-type")
P("  annotation is wrong and nothing in the myeloid blocks should be trusted:")
P("  " + (_pc[["gene", "arm_pair", "donor_median_a", "donor_median_b", "q", "status"]]
          .round(4).to_string(index=False).replace("\n", "\n  ") if len(_pc)
          else "not testable in this cohort."))

# ---------------------------------------------------------------- C
P("")
P(RULE)
P(f"C · LIGAND-RECEPTOR, CD4 <-> FIBROBLAST / MYELOID — {len(C_SPEC)} edges that are BOTH")
P(f"    significant (q<0.05) AND specific (spec_p<0.05), out of {len(C_SIG)} significant and "
  f"{int(C['q'].notna().sum())} tested")
P(RULE)
if len(C_SPEC):
    P(C_SPEC.sort_values(["direction", "spec_p", "q"])
      [["direction", "axis", "edge", "stroma_level", "median_delta", "pct_up", "n_pairs",
        "q", "spec_p_cd4_side", "spec_p_stroma_side", "spec_p", "spec_z", "higher"]]
      .rename(columns={"median_delta": "delta"}).round(4).to_string(index=False))
else:
    P("  none.")
P("")
P(f"  {len(C_SIG) - len(C_SPEC)} further edge x cell-type rows rose significantly in one arm")
P("  but did NOT beat both nulls. Those are abundance products, not specific pairs.")
_ctrl = C_SIG[C_SIG.edge.str.contains("S100A8|S100A9") & (C_SIG.higher == "malignant")]
P("  DIRECTION CONTROL — S100A8 and S100A9 are myeloid-restricted genes. If an edge carrying")
P("  them reads higher in the malignant CD4 arm, the cell-type labels are wrong: "
  + (", ".join(sorted(set(_ctrl.key))) + "  <-- LABELS ARE WRONG" if len(_ctrl) else "clean."))

# ---------------------------------------------------------------- D
P("")
P(RULE)
P(f"D · BURDEN -> STROMA ACROSS MF SAMPLES — {len(D_ROBUST)} robust rows (q<0.05 pooled AND "
  f"within study), {len(D_SIG)} significant pooled, of {int((Dt.status == 'tested').sum())} tested")
P(RULE)
P(f"Burden over {len(BURDEN)} MF samples: quantiles "
  f"{BURDEN['burden'].quantile([0, .25, .5, .75, 1]).round(2).to_dict()}.")
P("A correlation across samples cannot say which side drives the other, and burden rises with")
P("stage; read a robust row as 'this stromal state scales with tumour load'.")
_dcols = ["compartment", "stroma_row", "readout", "rho", "q", "rho_within_study",
          "q_within_study", "n_samples", "n_within", "n_studies"]
P("  robust:")
P(D_ROBUST[_dcols].round(4).to_string(index=False) if len(D_ROBUST) else "    none.")
_pooled_only = D_SIG[~D_SIG["robust"]]
if len(_pooled_only):
    P("  significant pooled but NOT within study (a between-study shift can explain these):")
    P(_pooled_only[_dcols].round(4).to_string(index=False))

# ---------------------------------------------------------------- abstentions
P("")
P(RULE)
P("9 · ABSTENTIONS — what this cohort could not be asked, kept apart from what it answered")
P(RULE)
P(f"levels below the donor gate ({cfg.MIN_CELLS} cells in {cfg.MIN_SAMPLES} donors): "
  f"{BELOW_GATE or 'none'}")
P(f"panel genes absent from the external reference ({len(EXT_MISSING)}): "
  f"{EXT_MISSING or 'none'}")
for tag in B_BLOCKS:
    d = B[(B.block == tag) & (B.status == "not_tested")]
    if not len(d):
        continue
    P(f"{tag}: {len(d)} untested rows -- "
      + "; ".join(f"{r} x{n}" for r, n in d["reason"].value_counts().items()))
P(f"A: {int(A['p'].isna().sum())} genes with fewer than {cfg.MIN_DONORS_PER_ARM} nonzero paired")
P("   differences -- never detected in either CD4 arm, or identical in every biopsy.")
P(f"C: {int(C['q'].isna().sum())} edge x cell-type rows below the paired-sample gate; "
  f"{len(lr_dropped)} curated panel entries could not be resolved to transcripts at all.")
P(f"D: {int((Dt.status == 'not_tested').sum())} row x readout pairs with < 10 samples or a")
P("   constant readout.")
P("")
P("MACHINE-READABLE TABLES BEHIND THIS REPORT: "
  + ", ".join(cfg.tab(n).name for n in
              ["A_paired_genes", "B_arm_tests", "C_ccc_edges", "D_burden_stroma", "burden",
               "ccc_pairs", "ccc_pairs_dropped", "coverage", "abstentions",
               "malignancy_concordance"]))
P("They carry every tested row, significant or not, with the same column names as above.")

txt = cfg.TAB_DIR / f"{cfg.TAB_PREFIX}results.txt"
txt.write_text("\n".join(L) + "\n")
D.append_run_log("NB28_mc — DEGRADOME, FOUR COMPARISONS ON mal_cluster",
                 "\n".join(str(r) for r in inv))
print("\n".join(L))
print(f"\n[full text also written to {txt}]")
```

```
==============================================================================================================
NB28_mc — DEGRADOME, FOUR COMPARISONS ON mal_cluster
==============================================================================================================
{'§': 'A', 'comparison': 'CD4, within biopsy', 'arms': 'malignant - reactive', 'replicates': 'n=88', 'tested': 126, 'significant': 84, 'not tested': 377}
{'§': 'A', 'comparison': 'CD4, within biopsy, ALICE-decided clusters', 'arms': 'malignant - reactive', 'replicates': 'n=30', 'tested': 77, 'significant': 30, 'not tested': 426}
{'§': 'B', 'comparison': 'CD4', 'arms': 'malignant / reactive / external HC', 'replicates': 'd=5-86', 'tested': 897, 'significant': 226, 'not tested': 0}
{'§': 'B', 'comparison': 'myeloid low res (4 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-84', 'tested': 2861, 'significant': 1026, 'not tested': 2059}
{'§': 'B', 'comparison': 'myeloid high res (12 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-67', 'tested': 6834, 'significant': 1762, 'not tested': 9690}
{'§': 'B', 'comparison': 'fibroblast low res (3 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-59', 'tested': 3778, 'significant': 1760, 'not tested': 2}
{'§': 'B', 'comparison': 'fibroblast high res (6 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-51', 'tested': 7041, 'significant': 2894, 'not tested': 897}
{'§': 'C', 'comparison': 'CD4 <-> fibro/myeloid LR edges', 'arms': 'malignant - reactive', 'replicates': 'n<=70', 'tested': 393, 'significant': 218, 'not tested': 1}
{'§': 'D', 'comparison': 'burden -> stromal state / composition', 'arms': 'Spearman across samples', 'replicates': 'n<=98', 'tested': 131, 'significant': 0, 'not tested': 51}
####################################################################################################
NB28_mc — THE CUTANEOUS DEGRADOME IN MYCOSIS FUNGOIDES SKIN, ON THE CLUSTER-LEVEL MALIGNANCY CALL
SELF-CONTAINED RESULTS REPORT — everything needed to interpret these numbers is below.
####################################################################################################

====================================================================================================
1 · THE QUESTION
====================================================================================================
Mycosis fungoides (MF) is a cutaneous T-cell lymphoma in which a malignant CD4 clone sits in
the skin alongside reactive T cells, myeloid cells and fibroblasts. The extracellular matrix
around it is remodelled. This notebook asks four things over a curated 'degradome' panel --
the proteases that dismantle each matrix compartment, their inhibitors, the matrix substrates,
and the adhesion/motility machinery a lymphocyte uses to move through what is left:
  (A) inside one biopsy, what does the malignant clone carry that its reactive neighbours do
      not?
  (B) across donors, how do MF cells of each type differ from healthy-control cells of the
      same type, in this atlas and in an independent healthy skin dataset?
  (C) which ligand-receptor edges between the clone and the stroma strengthen in the clone,
      specifically rather than as a by-product of abundance?
  (D) across MF samples, does the stroma track how much tumour the biopsy carries?

====================================================================================================
2 · THE DATA
====================================================================================================
Atlas v2, skin only, disease in ['MF', 'HC'] (MF patients and healthy controls) -- the scope
the degradome object was built on. SS and other CTCL entities are not in this object.
  985,273 cells | 112 donors | 123 samples | 12 studies
EXPRESSION UNITS: log1p(CP10K). The library size is summed over all 42,347 genes BEFORE the
  panel columns are subset, so values are comparable across cell types and sequencing
  depths. This is gated
... [1030087 more chars truncated]
```

# Part II — figures  (was `28b_tme_degradome_figures_mc`)

*Self-contained: run §0 below and this part alone; it reads Part I's `deg28mc_*` tables from disk.*

A fresh copy of `28b_tme_degradome_figures`. It reads **`28_tme_degradome_mc`**'s tables
(`tables/deg28mc_*`), and its malignant-vs-reactive CD4 arms come from **`mal_cluster`**
(`26b_cluster_malignancy`) instead of `mal_tcr`. 28b is left untouched for a later comparison.
Outputs are `figures/degradome/deg28bmc_*` and `tables/deg28bmc_*`.

| fig | question | source |
|---|---|---|
| F1 | degrading in malignant vs reactive vs healthy CD4; paired test run here | object |
| F2 | proteases in fibroblasts (a) and myeloid cells (b) beside the CD4 arms, MF skin | object |
| FB | T-cell → stroma ligands vs what the clone does carry (inhibitors, lysosomal cathepsins, granzymes) | object, recomputed |
| FC | LIGHT → LTβR/HVEM → RELB, read as a chain | object, recomputed |
| FD | Th2 axis: TF on the clone, cytokines, receptor and output on the stroma | object, recomputed |
| FE | hyaluronan: synthesis, fragmentation, CD44 — stroma (MF vs healthy) and CD4 | object, recomputed |
| FF | the TGF-β loop in fibroblasts beside the same genes in CD4 | object, recomputed |
| **FG** *(new)* | does the stroma track tumour burden across MF samples? | 28_mc §D + object |
| **\*_stage** *(new)* | each figure above split by stage: early vs advanced MF (`stage_class`) | same, per stage |

## Arms

The malignancy call is `mal_cluster`:
- each sample's T cells are Leiden-clustered, and each cluster is called malignant or benign
  (ALICE where the sample has TCR, else a gene program + V-gene skew, else a manual call);
- CD4 with no call are excluded;
- healthy CD4 from our own atlas are a **fourth arm** (`CD4_HC`), and HC is benign by construction.

**Circularity:** TOX (FD) is part of the program that calls `auto` clusters, so its
malignant-vs-reactive difference is partly built into the arms. It is marked on the figure. Read
28_mc §A's `*_tcr_src` columns for the program-free result.

## FA–FG statistics

- **Cohort.** Every MF arm is restricted to the MF samples with a `mal_cluster` call (~100,
  against 38 on `mal_tcr`), so the clone and the stroma come from the same biopsies. Healthy =
  atlas HC.
- **Replicate value.** Donor **mean** of log1p CP10K over ≥ `cfg.MIN_CELLS` cells.
- **Tests.**
  - Malignant vs reactive CD4: paired Wilcoxon signed-rank on the per-biopsy difference
    (`DC.paired_within`).
  - Otherwise: donor-level Mann-Whitney.
  - *Not testable* below `cfg.MIN_DETECTION` in both arms, or with fewer than
    `cfg.MIN_DONORS_PER_ARM` donors.
- **Correction.** BH over the tests drawn in each figure. These are not 28_mc's q-values.
- **Titles** are descriptive. **Captions** that state a finding are computed from these tests, or
  checked against them, because the mal_tcr run's claims need not hold under the new arms.
- **Stage (`*_stage`).** `stage_class`: early = IA–IIA, advanced = IIB+; MF samples of unknown
  stage are left out. Within a stage each figure re-runs its own test. Advanced vs early is
  pooled across studies: donor-level Mann-Whitney on donor means, and for the clone on the
  per-biopsy malignant − reactive difference averaged per donor. Early is mostly li2024; each
  caption prints the study mix.

## Outputs

`figures/degradome/deg28bmc_*.png` and `tables/deg28bmc_*.csv`. Nothing under `deg28_*`,
`deg28b_*` or `deg28mc_*` is written.


```python  # cell 12
# ============================================================================
# §0  Setup — the same preamble and the same build gate as nb28 §0
# ============================================================================
# Figures only. The gate still runs: if the size factor was not summed over all 42,347 genes
# before the panel subset, every value drawn below is scaled by a cell-type-dependent factor.
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
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

import degradome_data as cfg          # config only; frozen
import degradome_helpers as D         # build + gate
import degradome_cells as DC          # scores, tests, the plotting layer
import malignancy as MAL              # the three calls
import external_ref as X              # reynolds2021
import cluster_malig_helpers as CM    # the program that calls auto clusters

# Inputs are 28_mc's tables (cfg.tab -> deg28mc_*); this notebook writes only deg28bmc_*.
cfg.TAB_PREFIX = cfg.FIG_PREFIX = "deg28mc_"
PREFIX = "deg28bmc_"


def fig28b(name, ext="png"):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.FIG_DIR / f"{PREFIX}{name}.{ext}"


def tab28b(name):
    cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.TAB_DIR / f"{PREFIX}{name}.csv"


ARM_CALL = "mal_cluster"
CALL_PROGRAM = set(pd.read_csv(CM.PROGRAM_CSV)["gene"])
CLONAL, NONCLONAL = cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE
HC_CD4 = "CD4_HC"                     # healthy CD4 as an arm of its own; not an nb28 level

adata = D.load_degradome_adata()
eq = D.cached("equivalence",
              lambda: {"max_abs_diff": float(D.assert_build_equivalence(adata))})
assert eq["max_abs_diff"] < 1e-6, f"normalization gate failed: {eq}"
PANEL = [g for g in cfg.panel_genes() if g in set(map(str, adata.var_names))]

# 28_mc's tables. Nothing here recomputes them.
A = pd.read_csv(cfg.tab("A_paired_genes"))        # paired within biopsy, malignant vs reactive
B = pd.read_csv(cfg.tab("B_arm_tests"))           # donor-level, per block x row x gene x arm pair
CC = pd.read_csv(cfg.tab("C_ccc_edges"))          # curated ligand-receptor edges
COV = pd.read_csv(cfg.tab("coverage"))            # which levels may be spoken about

print(f"\nnormalization gate ok: max |lognorm(panel) - lognorm(full)| = {eq['max_abs_diff']:.3g}")
print(f"{adata.n_obs:,} cells | panel {len(PANEL)} genes | arms on {ARM_CALL}")
print(f"tables: A {len(A):,} genes | B {len(B):,} rows | C {len(CC):,} edge x level | "
      f"coverage {len(COV)} levels")
```

```
nb22 sidecar reconciled: 20 levels, 328,461 cells, zero drift
skin_degradome.h5ad: 985,273 cells x 503 genes, 112 donors, 123 samples, 12 studies
  scope ['MF', 'HC']: kept 1,002,481 of 1,345,527 skin cells; dropped 17,208 with no identity
  DEGRADOME-ONLY object: var is a hand-curated 503-gene panel. Not for HVG selection, PCA, UMAP, clustering or genome-wide DE. Use skin_degradome_pseudobulk_full.parquet for anything needing all genes.
[cache] equivalence <- equivalence.json

normalization gate ok: max |lognorm(panel) - lognorm(full)| = 0
985,273 cells | panel 503 genes | arms on mal_cluster
tables: A 503 genes | B 34,059 rows | C 394 edge x level | coverage 29 levels
```

```python  # cell 13
# ============================================================================
# §1  The arms — nb28 §2/§3, condensed, plus a healthy CD4 arm
# ============================================================================
# The object on disk was built with mal_knn as cfg.MALIG_HEADLINE. nb28 derives its own CD4
# split on mal_cluster and keeps the vocabulary identical, so writing that split straight into
# cfg.GROUPBY makes every plotting helper in degradome_cells -- all of which hardcode
# cfg.GROUPBY -- draw the mal_cluster arms with no helper edit.
mal = MAL.load_malignancy(adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs_names,
                          has_tcr=adata.obs.get("has_tcr"), verbose=False)
mal.index = adata.obs_names

IS_CD4 = (adata.obs[cfg.CELLTYPE_SRC].astype(str) == "CD4").to_numpy()
IS_MF = (adata.obs["disease"].astype(str) == "MF").to_numpy()
IS_HC = (adata.obs["disease"].astype(str) == cfg.HC_DISEASE).to_numpy()

LVL = pd.Series(adata.obs[cfg.GROUPBY].astype(str).to_numpy(), index=adata.obs_names)
LVL = LVL.mask(IS_CD4, pd.Series(MAL.cd4_level(adata.obs[cfg.CELLTYPE_SRC], mal,
                                               call=ARM_CALL).astype(str),
                                 index=adata.obs_names))
# Healthy CD4 is its own arm, not "reactive": malignant/reactive is a within-MF contrast and
# folding healthy cells into the reactive arm would make it mean two different things.
LVL = LVL.mask(IS_CD4 & IS_HC, HC_CD4)
adata.obs[cfg.GROUPBY] = pd.Categorical(LVL)
adata.obs["arm_level"] = adata.obs[cfg.GROUPBY]
LV = LVL.to_numpy()

# Stage for the *_stage companions: stage_class (early IA-IIA | advanced IIB+). stage_group needs
# an exact stage and calls gaydosik2019 / herrera2021 unknown although both cohorts are advanced.
STAGE_KEY = "stage_class"
STAGES = ["early", "advanced"]
STAGE_COL = {"early": "#e8a33d", "advanced": "#c0392b"}
STAGE = adata.obs[STAGE_KEY].astype(str).to_numpy()
_smp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
SAMPLE2DONOR = (pd.Series(adata.obs[cfg.DONOR_KEY].astype(str).to_numpy(), index=_smp)
                .groupby(level=0).first())
SAMPLE2STAGE = pd.Series(STAGE, index=_smp).groupby(level=0).first()

CLAIM = [lv for lv in cfg.CLAIM_LEVELS if lv in set(COV.loc[COV["claimable"], cfg.GROUPBY])]
FIBRO = [lv for lv in cfg.FIBRO_LEVELS if lv in CLAIM]
MYELOID = [lv for lv in cfg.MYELOID_LEVELS if lv in CLAIM]

ext = X.load_external()
EXT_T = ext.obs[X.TARGET_COL].astype(str).to_numpy()
ON_OBJ = set(LV)


def tme_rows(groups):
    """{row: {MF, HC, external HC}} for a {row_label: [fine levels]} map, plus per-row notes."""
    rows, notes = {}, {}
    for row, levels in groups.items():
        levels = [lv for lv in levels if lv in ON_OBJ]
        if not levels:
            continue
        m = np.isin(LV, levels)
        tg = cfg.EXT_ARM.get(row)
        rows[row] = {"MF": (adata, m & IS_MF), "HC": (adata, m & IS_HC),
                     "external HC": (ext, np.isin(EXT_T, tg)) if tg else None}
        note = []
        if row in cfg.EXT_POOLED:
            note.append(f"ext pooled: {'+'.join(tg)}")
        if not set(levels) & set(CLAIM):
            note.append("below donor gate")
        if note:
            notes[row] = "; ".join(note)
    return rows, notes


TME_ARMS = ["MF", "HC", "external HC"]
CD4_ARMS4 = ["malignant", "reactive", "HC CD4", "external HC"]
# DC.ARM_COLORS gives reactive and HC the same green; the healthy CD4 arm needs its own hue.
ARM_COLORS4 = {**DC.ARM_COLORS, "HC CD4": "#1f77b4"}

cd4_rows4 = {"CD4": {
    "malignant": (adata, (LV == CLONAL) & IS_MF),
    "reactive": (adata, (LV == NONCLONAL) & IS_MF),
    "HC CD4": (adata, LV == HC_CD4),
    "external HC": (ext, np.isin(EXT_T, cfg.EXT_ARM["CD4"])),
}}
mye_low, mye_low_notes = tme_rows(cfg.MYELOID_COARSE)
mye_high, mye_high_notes = tme_rows({lv: [lv] for lv in cfg.MYELOID_LEVELS})
fib_low, fib_low_notes = tme_rows(cfg.FIBRO_COARSE)
fib_high, fib_high_notes = tme_rows({lv: [lv] for lv in cfg.FIBRO_LEVELS})

FIB_FINE = [lv for lv in cfg.FIBRO_LEVELS if lv in ON_OBJ]
MYE_FINE = [lv for lv in cfg.MYELOID_LEVELS if lv in ON_OBJ]
STROMA_FINE = FIB_FINE + MYE_FINE

for lab, n in [("malignant", (LV == CLONAL) & IS_MF), ("reactive", (LV == NONCLONAL) & IS_MF),
               ("HC CD4", LV == HC_CD4), ("excluded, no call", (LV == cfg.CD4_UNASSESSED))]:
    print(f"  {lab}: {int(n.sum()):,} cells / "
          f"{adata.obs.loc[n, cfg.DONOR_KEY].nunique()} donors")
print(f"claimable {len(CLAIM)} levels | fibroblast {FIBRO} | myeloid {MYELOID}")
print(f"blocks: cd4 (1 row, 4 arms) | myeloid_low ({len(mye_low)}) | "
      f"myeloid_high ({len(mye_high)}) | fibro_low ({len(fib_low)}) | fibro_high ({len(fib_high)})")
```

```
malignant: 185,643 cells / 84 donors
  reactive: 137,664 cells / 86 donors
  HC CD4: 4,742 cells / 22 donors
  excluded, no call: 168 cells / 3 donors
claimable 18 levels | fibroblast ['F_papillary', 'F_reticular', 'F_mesenchymal', 'F_myofibroblast', 'F_inflammatory', 'F_apCAF'] | myeloid ['LC', 'cDC1', 'cDC2', 'DC_LAMP3', 'pDC', 'Mono_CD14', 'moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl']
blocks: cd4 (1 row, 4 arms) | myeloid_low (4) | myeloid_high (12) | fibro_low (3) | fibro_high (6)
```

```python  # cell 14
# ============================================================================
# F0  One MF biopsy on a UMAP: cell types, with the malignant CD4 cells (mal_cluster) marked
# ============================================================================
# The degradome object carries no embedding. The sample's rows are pulled from the skin MRVI-u
# latent (sample-unaware, row-aligned to joint_mrvi_input_skin.h5ad; only obs is read from that
# file), and neighbors + UMAP are run on that one sample. Sample = UMAP_SAMPLE, or else the MF
# biopsy with a mal_cluster call that covers the most cell types (>= 20 cells each), ties broken
# by min(malignant, reactive).
import h5py
import scanpy as sc
from anndata.io import read_elem

UMAP_SAMPLE = None                    # set a sample_id to override the automatic pick
SKIN_INPUT = NB_DIR / "data" / "atlas_joint" / "joint_mrvi_input_skin.h5ad"
MRVI_U = NB_DIR / "data" / "atlas_joint" / "joint_X_mrvi_u_skin.npy"

_s = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
_called = IS_MF & IS_CD4 & mal["mal_cluster_defined"].to_numpy()
_cand = pd.DataFrame({"sample": _s, "lv": LV})[np.isin(_s, np.unique(_s[_called]))]
_n = _cand.groupby(["sample", "lv"]).size().unstack(fill_value=0)
_rank = pd.DataFrame({"n_cells": _n.sum(1), "n_types": (_n >= 20).sum(1),
                      "n_mal": _n.get(CLONAL, 0), "n_react": _n.get(NONCLONAL, 0)})
_rank["balance"] = _rank[["n_mal", "n_react"]].min(1)
_rank = _rank[_rank["n_mal"] >= 20].sort_values(["n_types", "balance"], ascending=False)
SID = UMAP_SAMPLE or _rank.index[0]
print(_rank.head(5).to_string())

with h5py.File(SKIN_INPUT, "r") as f:
    _o = f["obs"]
    _ids = read_elem(_o["cell_id"] if "cell_id" in _o else _o[_o.attrs["_index"]])
_row = pd.Series(np.arange(len(_ids)), index=pd.Index(np.asarray(_ids).astype(str)))
_m = _s == SID
_cid = pd.Index(adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs_names)[_m]
_idx = _row.reindex(_cid.astype(str)).to_numpy()
assert not np.isnan(_idx).any(), f"{np.isnan(_idx).sum()} {SID} cells missing from the skin latent"
_idx = _idx.astype(int)
_srt = np.argsort(_idx)                                     # sorted rows: one pass over the mmap
_lat = np.empty((len(_idx), np.load(MRVI_U, mmap_mode="r").shape[1]), dtype=np.float32)
_lat[_srt] = np.load(MRVI_U, mmap_mode="r")[_idx[_srt]]

ad0 = sc.AnnData(obs=pd.DataFrame({"level": LV[_m]}, index=adata.obs_names[_m]))
ad0.obsm["X_mrvi_u"] = _lat
sc.pp.neighbors(ad0, use_rep="X_mrvi_u", n_neighbors=15, random_state=0)
sc.tl.umap(ad0, random_state=0)

_lab = {CLONAL: "Malignant CD4 (cluster call)", NONCLONAL: "Reactive CD4",
        cfg.CD4_UNASSESSED: "CD4, no call"}
_lvls = set(ad0.obs["level"])
_others = [lv for lv in cfg.CT_ORDER if lv in _lvls and lv not in (CLONAL, NONCLONAL)]
_others += sorted(_lvls - set(_others) - {CLONAL, NONCLONAL})
_cmap = plt.get_cmap("tab20")
_col = {lv: _cmap(i % 20) for i, lv in enumerate(_others)}
_col.update({CLONAL: DC.ARM_COLORS["clonal"], NONCLONAL: DC.ARM_COLORS["non-clonal"]})

U = ad0.obsm["X_umap"]
fig, ax = plt.subplots(figsize=(10, 6))
for lv in _others + [NONCLONAL, CLONAL]:                    # malignant drawn last, on top
    k = (ad0.obs["level"] == lv).to_numpy()
    if not k.any():
        continue
    hi = lv == CLONAL
    ax.scatter(U[k, 0], U[k, 1], s=4 if hi else 1.5, c=[_col[lv]], lw=0, rasterized=True,
               label=f"{_lab.get(lv, lv)} ({k.sum():,})", zorder=3 if hi else 2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
for _sp in ax.spines.values():
    _sp.set_visible(False)
_h, _l = ax.get_legend_handles_labels()
ax.legend(_h[::-1], _l[::-1], fontsize=5, markerscale=3, frameon=False, loc="center left",
          bbox_to_anchor=(1.0, 0.5), ncol=1 + (len(_l) > 14))
n_mal = int((ad0.obs["level"] == CLONAL).sum())
ax.set_title(f"{SID}: {n_mal:,} of {ad0.n_obs:,} cells are malignant CD4 (mal_cluster)",
             fontsize=8)
fig.savefig(fig28b("F0_umap_sample"), dpi=200, bbox_inches="tight")
plt.show()
print(ad0)

# F0a-c  The same UMAP three times: everything grey except one compartment.
def umap_highlight(levels, colors, title, tag):
    lv = ad0.obs["level"].to_numpy()
    on = np.isin(lv, levels)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(U[~on, 0], U[~on, 1], s=1, c="#d9d9d9", lw=0, rasterized=True, zorder=1,
               label=f"other ({(~on).sum():,})")
    for l in levels:
        k = lv == l
        if k.any():
            ax.scatter(U[k, 0], U[k, 1], s=3, c=[colors[l]], lw=0, rasterized=True, zorder=2,
                       label=f"{_lab.get(l, l)} ({k.sum():,})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    for _sp in ax.spines.values():
        _sp.set_visible(False)
    ax.legend(fontsize=6, markerscale=3, frameon=False, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"{SID}: {title}", fontsize=8)
    fig.savefig(fig28b(f"F0{tag}_umap_sample"), dpi=200, bbox_inches="tight")
    plt.show()


_fib = [l for l in FIB_FINE if l in _lvls]
_mye = [l for l in MYE_FINE if l in _lvls]
_tab = plt.get_cmap("tab10")
umap_highlight([CLONAL, NONCLONAL], _col,
               f"{n_mal:,} malignant vs {int((ad0.obs['level'] == NONCLONAL).sum()):,} "
               f"reactive CD4", "a")
umap_highlight(_fib, {l: _tab(i % 10) for i, l in enumerate(_fib)},
               f"{len(_fib)} fibroblast subtypes", "b")
umap_highlight(_mye, {l: plt.get_cmap("tab20")(i % 20) for i, l in enumerate(_mye)},
               f"{len(_mye)} myeloid subtypes", "c")
```

```
n_cells  n_types  n_mal  n_react  balance
sample                                                        
Li2024_atlas__CTCL8    34006       25   7379     2447     2447
Li2024_atlas__CTCL5    41011       24   7532     4073     4073
Li2024_atlas__CTCL7    77413       22   3658    38432     3658
D8__P163_LPD7_MF       10461       22    881      417      417
D7__P73_lesional       18145       22    189      344      189

<Figure size 1000x600 with 1 Axes>
AnnData object with n_obs × n_vars = 34006 × 0
    obs: 'level'
    uns: 'neighbors', 'umap'
    obsm: 'X_mrvi_u', 'X_umap'
    obsp: 'distances', 'connectivities'

<Figure size 1000x600 with 1 Axes>
<Figure size 1000x600 with 1 Axes>
<Figure size 1000x600 with 1 Axes>
```

```python  # cell 15
# ============================================================================
# §2  Scores, computed once
# ============================================================================
# Family score = plain mean of the family's genes on X (see the degradome_cells docstring for
# why not score_genes). Families below cfg.MIN_FAMILY_SIZE_FOR_SCORE get no mean and are read
# per gene instead -- ha_synthase and glycosidase are the two that matter here.
FAMS = DC.family_score_cells(adata)
BALS = DC.balance_cells(adata)
FAM_COLS = [f"fam_{k}" for k in FAMS]
BAL_COLS = [f"bal_{k}" for k in BALS]

# The replicate unit: one median per (level, donor), gated at cfg.MIN_CELLS.
DMED = DC.donor_median(adata, FAM_COLS + BAL_COLS)

# The external band. nb28 reads the medians the external build wrote rather than re-scoring
# 195,739 cells; a family whose gene set is not the same on both sides abstains entirely.
# The external build's artifacts keep nb28's deg28_ prefix; cfg.tab now points at deg28mc_.
EXT_DMED = X.load_external_donor_median(cfg.TAB_DIR / "deg28_external_donor_median.csv")
EXT_ABSTAIN, EXT_DROP = X.external_abstentions(cfg.TAB_DIR / "deg28_external_panel_coverage.csv")


def ext_band(scores, target):
    """{score: one value per external donor} for plot_split_violin / plot_external_band."""
    d = EXT_DMED[EXT_DMED["target"].astype(str) == target]
    return {s: d[s].dropna().to_numpy() for s in scores
            if s in EXT_DMED.columns and s not in set(EXT_DROP) and d[s].notna().any()}


# The paired slice: MF CD4 only, both arms, one row per (sample, arm). This is what §A tests
# and what plot_paired_slopes draws -- the sample is the replicate, so donor, study, chemistry,
# site and treatment are the same tube.
AXIS_CD4_GENES = ["TNFSF14", "LTA", "LTB", "CD40LG", "CXCR4", "CD44", "HYAL3", "HYAL2",
                  "IL7R", "SELL", "S1PR1", "ITGA4", "ITGA5", "ITGA6", "CCR7", "BSG"]
TOP_A = (A[A["q"].notna()].sort_values("q")["gene"].head(6).tolist())
SLICE_GENES = [g for g in dict.fromkeys(TOP_A + AXIS_CD4_GENES) if g in set(PANEL)]

cd4_mf = adata[IS_MF & np.isin(LV, [CLONAL, NONCLONAL])].copy()
SLICE_COLS = DC.gene_score_cells(cd4_mf, SLICE_GENES, verbose=False)
SM = DC.donor_median(cd4_mf, SLICE_COLS, groupby="arm_level", donor_key=cfg.SAMPLE_KEY)
N_PAIRS = int(SM[SM["eligible"]].pivot_table(index=cfg.SAMPLE_KEY, columns="arm_level",
                                             values=SLICE_COLS[0], aggfunc="median")
              .dropna().shape[0])

print(f"family scores: {len(FAM_COLS)} | balances: {len(BAL_COLS)} | "
      f"donor medians {DMED.shape} ({int(DMED['eligible'].sum()):,} eligible)")
print(f"external abstains: {EXT_ABSTAIN or 'none'}")
print(f"paired slice: {cd4_mf.n_obs:,} MF CD4 cells, {N_PAIRS} biopsies with both arms "
      f"above {cfg.MIN_CELLS} cells (28_mc §A reports {int(A['n_pairs'].max())})")
```

```python  # cell 16
# ============================================================================
# §3  Figure utilities — read nb28's tables, draw, never re-test
# ============================================================================
# B stores each pair as (arm_a, arm_b) with delta = donor_median_b - donor_median_a, so
# "MF vs HC" carries HC - MF. b_grid flips it to the direction a reader expects (arm_a minus
# arm_b) and keeps nb28's q untouched.
STAR = DC._stars
GREY = "#bdbdbd"


def b_grid(block, rows, genes, arm_pair="MF vs HC"):
    """(row x gene) donor medians, delta and q for one arm pair, straight out of deg28_B.

    `block` may be one block or several -- an axis read across fibroblasts and myeloid cells
    needs fibro_high and myeloid_high together, and their row names do not overlap.
    """
    blocks = [block] if isinstance(block, str) else list(block)
    d = B[B["block"].isin(blocks) & B["row"].isin(rows) & B["gene"].isin(genes)
          & (B["arm_pair"] == arm_pair)].copy()
    d["delta_ab"] = d["donor_median_a"] - d["donor_median_b"]     # arm_a minus arm_b
    return d


def heat(ax, d, rows, cols, value="delta_ab", row_col="row", col_col="gene", cmap="RdBu_r",
         vmax=None, title=None, cbar_label=None, star_col="q", annot=None, scale="shared"):
    """rows x cols heatmap with a BH star in each cell that has one. NaN cells are hatched.

    `scale="column"` gives every gene its own colour scale. Genes on this panel differ by an
    order of magnitude in absolute expression (CXCL12 moves by 1.8, CXCR4 by 0.2), and on one
    shared scale the smaller gene is a blank column -- which is not what the test says.
    """
    M = d.pivot_table(index=row_col, columns=col_col, values=value, aggfunc="mean")
    Q = d.pivot_table(index=row_col, columns=col_col, values=star_col, aggfunc="min")
    # rows/cols are kept as asked even when deg28_B has no row for them: a hatched cell says
    # "we could not look here", which an absent column does not.
    rows, cols = list(rows), list(cols)
    M, Q = M.reindex(index=rows, columns=cols), Q.reindex(index=rows, columns=cols)
    if scale == "column":
        cmax = np.nanmax(np.abs(M.to_numpy()), axis=0)
        cmax = np.where(np.isfinite(cmax) & (cmax > 0), cmax, 1.0)
        Z, v = M.to_numpy() / cmax, 1.0
    else:
        v = np.nanmax(np.abs(M.to_numpy())) if vmax is None else vmax
        v = 1.0 if not np.isfinite(v) or v == 0 else v
        Z = M.to_numpy()
    im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isfinite(M.iat[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True, color="#f0f0f0",
                                           hatch="///", edgecolor="white", linewidth=0))
                continue
            q = Q.iat[i, j]
            s = STAR(q) if np.isfinite(q) else ""
            if annot is not None:
                s = f"{M.iat[i, j]:{annot}}\n{s}" if s else f"{M.iat[i, j]:{annot}}"
            if s:
                ax.text(j, i, s, ha="center", va="center", fontsize=5.5,
                        color="#111111" if abs(Z[i, j]) < 0.6 * v else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6, style="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    ax.tick_params(length=2, pad=1)
    if title:
        ax.set_title(title, fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=5)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6)
    return M, Q


def arm_dots(ax, sl, arms, gene_idx, tests=None, row=None, gene=None, colors=None, ylabel=None):
    """One dot per donor per arm, from DC.arm_test_block's slices. The n a comparison has."""
    colors = colors or ARM_COLORS4
    rng = np.random.default_rng(0)
    tops = []
    for i, a in enumerate(arms):
        s = sl.get(a)
        if s is None:
            ax.text(i, 0, "no ext.\ncounterpart", ha="center", va="bottom", fontsize=5,
                    color="#888888", rotation=90)
            continue
        v = s["dmean"][:, gene_idx]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        tops.append(np.max(v))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, alpha=0.85,
                   color=colors.get(a, "#999999"), edgecolor="white", linewidth=0.3, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color="#111111", linewidth=1.2, zorder=4)
    top = max(tops) if tops else 1.0
    if tests is not None and row is not None and gene is not None:
        sig = tests[(tests["row"] == row) & (tests["gene"] == gene) & (tests["q"] < 0.05)]
        for k, t in enumerate(sorted(sig.itertuples(), key=lambda t: arms.index(t.arm_b)
                                     - arms.index(t.arm_a))):
            x0, x1 = arms.index(t.arm_a), arms.index(t.arm_b)
            h = top * (1.06 + 0.13 * k)
            ax.plot([x0, x0, x1, x1], [h - top * .03, h, h, h - top * .03], color="#333333",
                    linewidth=0.6, clip_on=False)
            ax.text((x0 + x1) / 2, h, STAR(t.q), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=90, fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel(ylabel or "", fontsize=6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return ax


def lollipop(ax, d, label_col="gene", value="pct_up", q_col="q", family_col="family",
             centre=0.5, xlabel=None, title=None, n_label=None):
    """One row per gene: dot at `value`, stem to `centre`, colour by family, grey if abstained.

    Direction is pct_up, not median_delta -- these panels are zero-inflated and the median
    paired delta is exactly 0 for most genes (nb28 §A).
    """
    d = d.reset_index(drop=True)
    fams = [f for f in dict.fromkeys(d[family_col].astype(str))]
    pal = DC._level_colors(fams)
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        v, q = r[value], r[q_col]
        ok = np.isfinite(v) and np.isfinite(q)
        c = pal[str(r[family_col])] if ok else GREY
        ax.plot([centre, v if ok else centre], [yi, yi], color=c, linewidth=1.0,
                alpha=0.8, zorder=2)
        ax.scatter([v if ok else centre], [yi], s=22 if ok else 10, color=c, zorder=3,
                   edgecolor="white", linewidth=0.4)
        lab = f"{r[label_col]}{' ' + STAR(q) if ok else ' (abstained)'}"
        ax.text(centre, yi, "  " + lab if (not ok or v >= centre) else lab + "  ",
                ha="left" if (not ok or v >= centre) else "right", va="center", fontsize=5.5,
                style="italic", zorder=4)
    ax.axvline(centre, color="#333333", linewidth=0.8, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-1, len(d))
    ax.tick_params(axis="x", labelsize=6)
    if xlabel or n_label:
        ax.set_xlabel("\n".join([x for x in (xlabel, n_label) if x]), fontsize=6)
    if title:
        ax.set_title(title, fontsize=8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=pal[f], label=f,
                          markersize=4) for f in fams]
    return handles


def fam_arm_test(dmed, levels, score_cols, group="disease", arm_a="MF", arm_b="HC"):
    """Donor-level MWU on the FAMILY score, arm_a vs arm_b. BH over this grid only.

    nb28's tested unit is the gene, so a family-level star does not exist in deg28_B and has to
    be computed here. Same replicate (the donor), same test, same gate as DC.arm_tests.
    """
    out = []
    for lv in levels:
        d = dmed[(dmed[cfg.GROUPBY].astype(str) == lv) & dmed["eligible"]]
        for s in score_cols:
            x = d.loc[d[group].astype(str) == arm_a, s].dropna().to_numpy()
            y = d.loc[d[group].astype(str) == arm_b, s].dropna().to_numpy()
            p = np.nan
            if (len(x) >= cfg.MIN_DONORS_PER_ARM and len(y) >= cfg.MIN_DONORS_PER_ARM
                    and np.ptp(np.r_[x, y]) > 0):
                p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            out.append({"row": lv, "gene": s, "n_donors_a": len(x), "n_donors_b": len(y),
                        "donor_median_a": np.median(x) if len(x) else np.nan,
                        "donor_median_b": np.median(y) if len(y) else np.nan, "p": p})
    r = pd.DataFrame(out)
    r["delta_ab"] = r["donor_median_a"] - r["donor_median_b"]
    r["q"] = np.nan
    m = r["p"].notna()
    if m.any():
        r.loc[m, "q"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


FIG_INDEX = []


def logfig(fig_id, path, question, source, note=""):
    FIG_INDEX.append({"figure": fig_id, "file": Path(path).name, "question": question,
                      "source": source, "note": note})
    print(f"  {fig_id} -> {Path(path).name}")


print("figure utilities ready")
```

```python  # cell 17
# ============================================================================
# F1  Degrading in malignant vs reactive vs healthy CD4
# ============================================================================
# Two panels because they answer the question at two grains: the family score says how much
# degradative machinery a CD4 cell carries at all, the gene grid says which enzyme. The paired
# malignant-vs-reactive test (28_mc §A design) runs here on this call's arms; its one-line summary
# is printed, and the titles only name the figure.
F1_SCORES = [s for s in ["fam_mmp", "fam_cathepsin", "fam_serine_other", "fam_metallo_other",
                         "fam_granzyme", "fam_ha_catabolic", "fam_inhibitor",
                         "fam_matrix_receptor"] if s in DMED.columns]
CD4_LEVELS3 = [CLONAL, NONCLONAL, HC_CD4]
# The headline proteases only: the four MMPs CD4 cells express at all, the canonical sheddases
# ADAM10/17 plus the leukocyte ADAM8/19, uPAR and heparanase.
F1_GENES = [g for g in ["MMP2", "MMP9", "MMP14", "MMP25",
                        "ADAM8", "ADAM10", "ADAM17", "ADAM19",
                        "PLAUR", "HPSE"] if g in set(PANEL)]


def paired_title(cols, names, what, obj=None):
    """Summary + table from the paired within-biopsy test of `cols` on the MF CD4 arms (`obj`,
    default all of cd4_mf)."""
    sm = DC.donor_median(cd4_mf if obj is None else obj, cols, groupby="arm_level",
                         donor_key=cfg.SAMPLE_KEY)
    r = DC.paired_within(sm, cols, pair_key=cfg.SAMPLE_KEY, arm_col="arm_level",
                         ref=NONCLONAL, alt=CLONAL, fdr=True)
    r["name"] = r["score"].map(names)
    sig_ = r["q"] < 0.05
    up = r.loc[sig_ & (r["n_pos"] > r["n_neg"]), "name"].tolist()
    dn = r.loc[sig_ & (r["n_pos"] < r["n_neg"]), "name"].tolist()
    head = (f"Higher in malignant CD4: {', '.join(up)}" if up
            else f"No {what} is higher in malignant than reactive CD4")
    tail = f"; higher in reactive: {', '.join(dn)}" if dn else ""
    return f"{head}{tail} (paired within biopsy, {int(r['n_pairs'].max())} biopsies, BH)", r


F1A_TITLE, F1A_PAIRED = paired_title(F1_SCORES, {s: s.replace("fam_", "") for s in F1_SCORES},
                                     "degradome family")
_gcols = DC.gene_score_cells(cd4_mf, F1_GENES, verbose=False)
F1B_TITLE, F1B_PAIRED = paired_title(_gcols, dict(zip(_gcols, F1_GENES)), "headline protease")
cd4_mf.obs.drop(columns=[c for c in _gcols if c not in SLICE_COLS], inplace=True)
pd.concat([F1A_PAIRED.assign(panel="F1a"), F1B_PAIRED.assign(panel="F1b")]).to_csv(
    tab28b("F1_paired_tests"), index=False)

DC.plot_split_violin(
    adata, DMED, F1_SCORES, CD4_LEVELS3,
    title=("Degradome family scores in malignant, reactive and healthy CD4\n"
           "violin = cells, tick = median, dots = donors (≥ 25 cells); "
           "band = reynolds2021 healthy CD4, 5 donors"),
    external=ext_band(F1_SCORES, "CD4"), ncols=4,
    save=fig28b("F1a_cd4_family_scores"))
logfig("F1a", fig28b("F1a_cd4_family_scores"),
       "degradome family load, malignant vs reactive vs healthy CD4",
       "object (donor medians) + deg28_external_donor_median; paired test in F1_paired_tests")

f1b = DC.plot_family_arms(
    cd4_rows4, F1_GENES, CD4_ARMS4, arm_colors=ARM_COLORS4,
    title="Headline protease genes in malignant, reactive and healthy CD4",
    wrap=5, col_w=2.2, row_h=2.6, expressing_only=True, clean=True,
    save=fig28b("F1b_cd4_protease_genes"))
f1b.to_csv(tab28b("F1_source"), index=False)
logfig("F1b", fig28b("F1b_cd4_protease_genes"),
       "which protease, if any, the clone carries",
       "object; donor-level MWU recomputed over the drawn genes; paired test in F1_paired_tests",
       note="q recomputed figure-wide: the 4-arm set includes atlas HC CD4")
print(F1A_TITLE)
print(F1B_TITLE)
```

```python  # cell 18
# ============================================================================
# F1_stage  F1 split by stage: early vs advanced MF
# ============================================================================
# Stage = stage_class; MF samples of unknown stage are left out. Within a stage, F1's paired
# within-biopsy test runs on that stage's biopsies. Advanced vs early is pooled across studies: a
# donor-level Mann-Whitney on the per-biopsy malignant - reactive difference, averaged per donor,
# so study and chemistry cancel inside the difference.
_cd4_stage = cd4_mf.obs[STAGE_KEY].astype(str).to_numpy()


def stage_gap_test(cols, names):
    """Advanced vs early on the per-biopsy (median) malignant - reactive gap, donor-level MWU."""
    sm = DC.donor_median(cd4_mf, cols, groupby="arm_level", donor_key=cfg.SAMPLE_KEY)
    sm = sm[sm["eligible"]]
    out = []
    for c in cols:
        w = (sm.pivot_table(index=cfg.SAMPLE_KEY, columns="arm_level", values=c, aggfunc="median")
             .reindex(columns=[CLONAL, NONCLONAL]).dropna())
        donor = SAMPLE2DONOR.reindex(w.index).to_numpy()
        gap = (w[CLONAL] - w[NONCLONAL]).groupby(donor).mean()
        st = pd.Series(SAMPLE2STAGE.reindex(w.index).to_numpy()).groupby(donor).first()
        x, y = gap[st == "advanced"].to_numpy(), gap[st == "early"].to_numpy()
        p = (float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
             if min(len(x), len(y)) >= cfg.MIN_DONORS_PER_ARM and np.ptp(np.r_[x, y]) > 0
             else np.nan)
        out.append({"score": c, "name": names[c], "n_advanced": len(x), "n_early": len(y),
                    "gap_advanced": np.median(x) if len(x) else np.nan,
                    "gap_early": np.median(y) if len(y) else np.nan, "p": p})
    r = pd.DataFrame(out)
    r["q"] = np.nan
    m = r["p"].notna()
    if m.any():
        r.loc[m, "q"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


def stage_title(cols, names, what):
    """F1's paired-test summary per stage, plus the advanced-vs-early gap line; and the tests."""
    lines, tabs = [], []
    for st in STAGES:
        t, r = paired_title(cols, names, what, obj=cd4_mf[_cd4_stage == st])
        lines.append(f"{st}: {t}")
        tabs.append(r.assign(stage=st))
    g = stage_gap_test(cols, names)
    ch = g[g["q"] < 0.05]
    moved = ", ".join(f"{r['name']} {'higher' if r['gap_advanced'] > r['gap_early'] else 'lower'}"
                      for _, r in ch.iterrows())
    lines.append(f"malignant - reactive gap, advanced vs early: {moved or 'no change'} "
                 f"(MWU on donor gap, {int(g['n_early'].max())} early | "
                 f"{int(g['n_advanced'].max())} advanced donors, BH)")
    return "\n".join(lines), pd.concat(tabs + [g.assign(stage="advanced vs early")])


# F1a_stage: the malignant and reactive arms of each stage side by side, healthy CD4 last.
F1S_NAME = {(CLONAL, "early"): "malignant early", (NONCLONAL, "early"): "reactive early",
            (CLONAL, "advanced"): "malignant adv.", (NONCLONAL, "advanced"): "reactive adv."}
_arm = np.full(adata.n_obs, "other", dtype=object)
for (lv, st), name in F1S_NAME.items():
    _arm[(LV == lv) & IS_MF & (STAGE == st)] = name
_arm[LV == HC_CD4] = "healthy CD4"
adata.obs["arm_stage"] = pd.Categorical(_arm)
F1S_LEVELS = list(F1S_NAME.values()) + ["healthy CD4"]
DMED_S = DC.donor_median(adata, F1_SCORES, groupby="arm_stage",
                         extra=("study", STAGE_KEY, "disease"))

F1AS_TITLE, F1AS_TESTS = stage_title(F1_SCORES, {s: s.replace("fam_", "") for s in F1_SCORES},
                                     "degradome family")
_gcols = DC.gene_score_cells(cd4_mf, F1_GENES, verbose=False)
F1BS_TITLE, F1BS_TESTS = stage_title(_gcols, dict(zip(_gcols, F1_GENES)), "headline protease")
cd4_mf.obs.drop(columns=[c for c in _gcols if c not in SLICE_COLS], inplace=True)
pd.concat([F1AS_TESTS.assign(panel="F1a"), F1BS_TESTS.assign(panel="F1b")]).to_csv(
    tab28b("F1_stage_tests"), index=False)

DC.plot_split_violin(
    adata, DMED_S, F1_SCORES, F1S_LEVELS, level_col="arm_stage",
    title=("Degradome family scores in malignant, reactive and healthy CD4, early and advanced MF\n"
           "violin = cells, tick = median, dots = donors (≥ 25 cells); stage = stage_class, "
           "unknown stage left out; band = reynolds2021 healthy CD4, 5 donors"),
    external=ext_band(F1_SCORES, "CD4"), ncols=4, panel=(2.4, 3.2),
    save=fig28b("F1a_stage_cd4_family_scores"))
logfig("F1a_stage", fig28b("F1a_stage_cd4_family_scores"),
       "degradome family load in malignant vs reactive CD4, early and advanced MF",
       "object (donor medians); paired test per stage + donor MWU on the gap (F1_stage_tests)")

# F1b_stage: F1b's four arms once per stage; plot_family_arms tests arm pairs within a row only.
cd4_rows_stage = {f"{st} MF": {
    "malignant": (adata, (LV == CLONAL) & IS_MF & (STAGE == st)),
    "reactive": (adata, (LV == NONCLONAL) & IS_MF & (STAGE == st)),
    "HC CD4": (adata, LV == HC_CD4),
    "external HC": (ext, np.isin(EXT_T, cfg.EXT_ARM["CD4"])),
} for st in STAGES}
f1bs = DC.plot_family_arms(
    cd4_rows_stage, F1_GENES, CD4_ARMS4, arm_colors=ARM_COLORS4,
    title="Headline protease genes in malignant, reactive and healthy CD4, early and advanced MF",
    wrap=5, col_w=2.2, row_h=2.6, expressing_only=True, clean=True,
    save=fig28b("F1b_stage_cd4_protease_genes"))
f1bs.to_csv(tab28b("F1_stage_source"), index=False)
logfig("F1b_stage", fig28b("F1b_stage_cd4_protease_genes"),
       "which protease the clone carries, early and advanced MF",
       "object; donor-level MWU within each stage row; paired test per stage (F1_stage_tests)",
       note="q recomputed figure-wide; healthy and external arms shared by both rows")
print(F1AS_TITLE)
print(F1BS_TITLE)
```

```python  # cell 19
# ============================================================================
# F2  Where the proteases are: fibroblasts and myeloid cells next to the CD4 arms, MF skin
# ============================================================================
# The two CD4 arms stay in both panels -- the contrast is the claim: near-empty CD4 violins for
# every protease beside stromal violins that are not. MF samples only; dots are donor medians
# (>= cfg.MIN_CELLS cells), the donor being the replicate. No test is drawn.
F2_CD4 = {"CD4_clonal": (LV == CLONAL) & IS_MF, "CD4_non_clonal": (LV == NONCLONAL) & IS_MF}
F2_PANELS = {
    "F2a": ("fibro",
            {"F_structural": np.isin(LV, cfg.FIBRO_COARSE["F_structural"]) & IS_MF,
             "F_inflammatory": (LV == "F_inflammatory") & IS_MF,
             "F_apCAF": (LV == "F_apCAF") & IS_MF},
            ["MMP2", "MMP14", "ADAMTS5", "CTSB", "CTSK", "PRSS23", "TIMP1", "TIMP3"],
            "Protease genes in fibroblasts vs malignant and reactive CD4, MF skin"),
    "F2b": ("myeloid",
            {lv: (LV == lv) & IS_MF for lv in ["Mac_FOLR2", "Mono_CD14", "cDC2"]},
            ["MMP12", "MMP9", "MMP14", "CTSB", "CTSC", "CTSS", "PLAUR", "SERPINA1"],
            "Protease genes in myeloid cells vs malignant and reactive CD4, MF skin"),
}
F2_COLORS = {"CD4_clonal": DC.ARM_COLORS["clonal"], "CD4_non_clonal": DC.ARM_COLORS["non-clonal"],
             "F_structural": "#08519c", "F_inflammatory": "#3182bd", "F_apCAF": "#9ecae1",
             "Mac_FOLR2": "#a63603", "Mono_CD14": "#e6550d", "cDC2": "#fdae6b"}
# Display names on the figure only; the source tables keep the atlas labels (methods).
F2_LABELS = {"CD4_clonal": "Malignant CD4 (cluster call)",
             "CD4_non_clonal": "Reactive CD4 (same biopsy)",
             "F_structural": "Structural fibroblast",
             "F_inflammatory": "Inflammatory fibroblast",
             "F_apCAF": "MHC-II+ fibroblast (apCAF)",
             "Mac_FOLR2": "Resident macrophage (FOLR2+)",
             "Mono_CD14": "Classical monocyte (CD14+)",
             "cDC2": "Conventional DC2"}

for tag, (lineage, stroma, genes, title) in F2_PANELS.items():
    d = DC.plot_group_violins(adata, {**F2_CD4, **stroma}, genes, title=title,
                              colors=F2_COLORS, labels=F2_LABELS,
                              save=fig28b(f"{tag}_{lineage}_vs_cd4"))
    d.to_csv(tab28b(f"{tag}_source_{lineage}"), index=False)
    logfig(tag, fig28b(f"{tag}_{lineage}_vs_cd4"),
           f"protease genes per cell type, CD4 arms vs {lineage}, MF skin",
           "object (per-cell log1p CP10K; donor medians)")
```

```python  # cell 20
# ============================================================================
# F2_stage  F2 split by stage: the same cell types in early and in advanced MF skin
# ============================================================================
# Descriptive, as F2: no test. Early block first, then advanced. The printed line counts, per
# stage, the proteases for which some stromal type's median donor mean is above both CD4 arms'.
from matplotlib.colors import to_rgb

_VAR = pd.Index(map(str, adata.var_names))
_DONOR = adata.obs[cfg.DONOR_KEY].astype(str).to_numpy()


def median_donor_mean(m, genes):
    """Median over donors (>= cfg.MIN_CELLS cells) of the donor mean log1p CP10K, per gene."""
    idx = np.flatnonzero(m)
    Xs = adata.X[idx][:, _VAR.get_indexer(genes)]
    Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    g = pd.DataFrame(Xs, columns=genes).groupby(_DONOR[idx])
    return g.mean()[(g.size() >= cfg.MIN_CELLS).to_numpy()].median()


def pale(c, f=0.45):
    """`c` mixed toward white: the early block's shade of a cell type's colour."""
    return tuple(1 - f * (1 - v) for v in to_rgb(c))


F2S_WHO = {"fibro": "Fibroblasts", "myeloid": "Myeloid cells"}
for tag, (lineage, stroma, genes, _) in F2_PANELS.items():
    arms = {**F2_CD4, **stroma}
    groups = {f"{k}·{st}": m & (STAGE == st) for st in STAGES for k, m in arms.items()}
    med = pd.DataFrame({k: median_donor_mean(m, genes) for k, m in groups.items()}).T
    wins = {st: sum(bool(med.loc[[f"{k}·{st}" for k in stroma], g].max()
                         > med.loc[[f"{k}·{st}" for k in F2_CD4], g].max()) for g in genes)
            for st in STAGES}
    title = (f"Protease genes in {F2S_WHO[lineage].lower()} vs malignant and reactive CD4, "
             "early and advanced MF skin\n"
             "median of donor means; left block early (pale), right block advanced; "
             "stage = stage_class, unknown stage left out")
    d = DC.plot_group_violins(
        adata, groups, genes, title=title,
        colors={f"{k}·{st}": pale(F2_COLORS[k]) if st == "early" else F2_COLORS[k]
                for st in STAGES for k in arms},
        labels={f"{k}·{st}": f"{F2_LABELS[k]} · {st}" for st in STAGES for k in arms},
        panel=(3.8, 2.4), save=fig28b(f"{tag}_stage_{lineage}_vs_cd4"))
    d.to_csv(tab28b(f"{tag}_stage_source_{lineage}"), index=False)
    logfig(f"{tag}_stage", fig28b(f"{tag}_stage_{lineage}_vs_cd4"),
           f"protease genes per cell type, CD4 arms vs {lineage}, early and advanced MF skin",
           "object (per-cell log1p CP10K; donor medians)")
    print(f"{F2S_WHO[lineage]} are above both CD4 arms for {wins['early']}/{len(genes)} "
          f"proteases in early and {wins['advanced']}/{len(genes)} in advanced MF skin")
```

```python  # cell 21
# ============================================================================
# §4  FA–FE layer — mal_cluster cohort, donor means, every test recomputed here
# ============================================================================
# Every MF arm below is restricted to the MF samples whose CD4 cells carry a mal_cluster call,
# the sample set 28_mc §A and §C rest on -- so clone and stroma come from the same biopsies.
# The replicate value is the donor MEAN of log1p CP10K: these genes are zero-inflated and a
# per-cell median is exactly 0 in most donors. Both choices move the numbers, so no star here is
# nb28's q: each is recomputed on donor means and BH-corrected within its own figure.
import textwrap

import scipy.sparse as sp
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import offset_copy

_samp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
CALL_SAMPLES = sorted(set(_samp[IS_MF & IS_CD4 & mal["mal_cluster_defined"].to_numpy()]))
IS_MF_CALL = IS_MF & np.isin(_samp, CALL_SAMPLES)
VAR = pd.Index(map(str, adata.var_names))

# Display names on the figures only; source tables keep the atlas labels (methods).
LABELS = {CLONAL: "Malignant CD4 (cluster call)", NONCLONAL: "Reactive CD4 (same biopsy)",
          HC_CD4: "Healthy-donor CD4",
          "F_structural": "Structural fibroblast", "F_inflammatory": "Inflammatory fibroblast",
          "F_apCAF": "MHC-II+ fibroblast (apCAF)", "F_reticular": "Reticular fibroblast",
          "F_mesenchymal": "Mesenchymal fibroblast",
          "Mac_FOLR2": "Resident macrophage (FOLR2+)", "Mono_CD14": "Classical monocyte (CD14+)",
          "cDC2": "Conventional DC2"}
UP, DOWN, NS = "#2a9d8f", "#b5542a", "#bdbdbd"
DIR_COLOR = {"up": UP, "down": DOWN, "ns": NS, "not tested": NS}
MALIG, REACT, HC_COL = "#7b3294", "#5e81ac", "#444444"
ARM_COL = {CLONAL: MALIG, NONCLONAL: REACT, HC_CD4: HC_COL}
XLAB = "donor mean, log1p CP10K"


def mask(ct, arm):
    """Cells of one cell type in one arm: 'MF' = called MF samples, 'HC' = atlas healthy skin,
    'early' / 'advanced' = the called MF samples of that stage (the *_stage figures).

    The healthy arm of either CD4 arm is healthy-donor CD4. F_structural is the coarse pool
    (papillary + reticular + mesenchymal + myofibroblast), as in F2.
    """
    if ct == HC_CD4 or (arm == "HC" and ct in (CLONAL, NONCLONAL)):
        return LV == HC_CD4
    m = np.isin(LV, cfg.FIBRO_COARSE.get(ct, [ct]))
    if arm == "HC":
        return m & IS_HC
    return m & IS_MF_CALL & (STAGE == arm if arm in STAGES else True)


_DM = {}


def donor_means(ct, arm, genes, key=cfg.DONOR_KEY):
    """(donor x gene) mean log1p CP10K, donors with >= cfg.MIN_CELLS cells; plus detection.

    Cached per (ct, arm, key); genes not yet cached are added on demand. `key=cfg.SAMPLE_KEY`
    gives the per-biopsy means the paired test runs on.
    """
    k = (ct, arm, key)
    genes = list(dict.fromkeys(genes))
    need = [g for g in genes if k not in _DM or g not in _DM[k][0].columns]
    if need:
        idx = np.flatnonzero(mask(ct, arm))
        Xs = adata.X[idx][:, VAR.get_indexer(need)]
        Xs = Xs.tocsr() if sp.issparse(Xs) else sp.csr_matrix(Xs)
        codes, uniq = pd.factorize(adata.obs[key].astype(str).to_numpy()[idx])
        cnt = np.bincount(codes, minlength=len(uniq))
        G = sp.csr_matrix((np.ones(len(idx)), (codes, np.arange(len(idx)))),
                          shape=(len(uniq), len(idx)))
        M = pd.DataFrame(np.asarray((G @ Xs).todense()) / np.maximum(cnt, 1)[:, None],
                         index=pd.Index(uniq, name=key), columns=need)
        det = pd.Series(np.asarray((Xs > 0).mean(axis=0)).ravel() if len(idx) else np.nan,
                        index=need)
        n = pd.Series(cnt, index=M.index)
        if k in _DM:
            M, det = _DM[k][0].join(M), pd.concat([_DM[k][1], det])
        _DM[k] = (M, det, n.reindex(M.index))
    M, det, cnt = _DM[k]
    return M.loc[cnt >= cfg.MIN_CELLS, genes], det[genes]


def prefetch(cts, genes, arms=("MF", "HC")):
    for ct in cts:
        for arm in arms:
            donor_means(ct, arm, genes)


def cell_values(ct, genes, max_cells=20_000, arm="MF"):
    """Per-cell log1p CP10K in one MF arm, subsampled for the violin KDE only (seed fixed)."""
    idx = np.flatnonzero(mask(ct, arm))
    if len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_cells, replace=False))
    Xs = adata.X[idx][:, VAR.get_indexer(genes)]
    return pd.DataFrame(Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs), columns=genes)


# ---------------------------------------------------------------- tests
def t_mwu(ct, gene, a="MF", b="HC", ct_b=None, label=None):
    """Donor-level Mann-Whitney on donor means: ct in arm a vs (ct_b or ct) in arm b."""
    ct_b = ct_b or ct
    ma, da = donor_means(ct, a, [gene])
    mb, db = donor_means(ct_b, b, [gene])
    x, y = ma[gene].to_numpy(), mb[gene].to_numpy()
    rec = {"label": label or LABELS.get(ct, ct), "gene": gene, "ct": ct, "arm_a": a,
           "ct_b": HC_CD4 if (b == "HC" and ct_b in (CLONAL, NONCLONAL)) else ct_b, "arm_b": b,
           "test": "MWU (donor)", "n_a": len(x), "n_b": len(y),
           "med_a": np.median(x) if len(x) else np.nan,
           "med_b": np.median(y) if len(y) else np.nan,
           "det_a": da[gene], "det_b": db[gene], "p": np.nan, "reason": ""}
    rec["delta"] = rec["med_a"] - rec["med_b"]
    rec["sign"] = np.sign(rec["delta"])
    if not np.nanmax([rec["det_a"], rec["det_b"]]) >= cfg.MIN_DETECTION:
        rec["reason"] = f"below {cfg.MIN_DETECTION:.0%} detection in both arms"
    elif min(len(x), len(y)) < cfg.MIN_DONORS_PER_ARM:
        rec["reason"] = f"too few donors ({len(x)}, {len(y)}; need {cfg.MIN_DONORS_PER_ARM})"
    elif np.ptp(np.r_[x, y]) == 0:
        rec["reason"] = "no variance"
    else:
        rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    rec["vals_a"], rec["vals_b"] = x, y
    return rec


def t_paired(genes, label="vs Reactive CD4 (same biopsy)", arm="MF"):
    """Clone vs reactive, paired within biopsy: nb28 §A's signed-rank, run on per-biopsy means.

    DC.paired_within is the §A function itself; only the per-biopsy summary changes (mean, not
    median), and the cohort is the same by construction (both arms need a mal_cluster call).
    `arm` = "MF", or one stage to pair only that stage's biopsies.
    """
    genes = list(dict.fromkeys(genes))
    a, da = donor_means(CLONAL, arm, genes, key=cfg.SAMPLE_KEY)
    b, db = donor_means(NONCLONAL, arm, genes, key=cfg.SAMPLE_KEY)
    long = (pd.concat([a.assign(arm=CLONAL), b.assign(arm=NONCLONAL)])
            .reset_index().assign(eligible=True))
    res = DC.paired_within(long, genes, pair_key=cfg.SAMPLE_KEY, arm_col="arm",
                           ref=NONCLONAL, alt=CLONAL, fdr=False).set_index("score")
    both = a.index.intersection(b.index)
    out = []
    for g in genes:
        r = res.loc[g]
        x, y = a.loc[both, g].to_numpy(), b.loc[both, g].to_numpy()
        det_ok = np.nanmax([da[g], db[g]]) >= cfg.MIN_DETECTION
        out.append({"label": label, "gene": g, "ct": CLONAL, "arm_a": arm, "ct_b": NONCLONAL,
                    "arm_b": arm, "test": "paired signed-rank (biopsy)",
                    "n_a": len(x), "n_b": len(y),
                    "med_a": np.median(x) if len(x) else np.nan,
                    "med_b": np.median(y) if len(y) else np.nan,
                    "det_a": da[g], "det_b": db[g],
                    "p": r["p"] if det_ok else np.nan,
                    "reason": r["reason"] if det_ok
                    else f"below {cfg.MIN_DETECTION:.0%} detection in both arms",
                    "delta": r["median_delta"], "sign": np.sign(r["n_pos"] - r["n_neg"]),
                    "n_pairs": r["n_pairs"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                    "vals_a": x, "vals_b": y})
    return out


def bh(recs):
    """BH over every test drawn in one figure; direction = sign of a vs b where q < 0.05."""
    df = pd.DataFrame(recs).reset_index(drop=True)
    m = df["p"].notna()
    df["q"] = np.nan
    if m.any():
        df.loc[m, "q"] = multipletests(df.loc[m, "p"], method="fdr_bh")[1]
    df["direction"] = np.select([~m, df["q"] >= 0.05, df["sign"] > 0],
                                ["not tested", "ns", "up"], "down")
    return df


def save_source(df, name):
    df.drop(columns=["vals_a", "vals_b"], errors="ignore").to_csv(tab28b(name), index=False)


def show_tests(df, cols=("label", "gene", "n_a", "n_b", "med_a", "med_b", "q", "direction")):
    print(df[[c for c in cols if c in df.columns]].to_string(index=False, float_format="%.3g"))


# ---------------------------------------------------------------- drawing
def it(g):
    return rf"$\mathit{{{g}}}$"


def pct(x):
    if not np.isfinite(x):
        return "–"
    return f"{x:.0%}" if x >= 0.01 else f"{x:.1%}" if x >= 0.001 else "<0.1%"


def sig(q):
    return "n.t." if not np.isfinite(q) else (STAR(q) or "n.s.")


def _clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=6, length=2)


def _violin(ax, v, pos, color, width=0.36, horizontal=False):
    v = np.asarray(v)[np.isfinite(v)]
    if len(v) and np.ptp(v) > 0:
        b = ax.violinplot([v], positions=[pos], widths=width, showextrema=False,
                          orientation="horizontal" if horizontal else "vertical")["bodies"][0]
        b.set_facecolor(color)
        b.set_alpha(0.3)
        b.set_edgecolor("none")


def _layout(df, group_col=None):
    """y per row (first row on top) plus one bold header line per group."""
    ys, heads, k, prev = [], [], 0, object()
    for g in (df[group_col] if group_col else [None] * len(df)):
        if group_col and g != prev:
            heads.append((k, g))
            k, prev = k + 1, g
        ys.append(k)
        k += 1
    return np.array([k - 1 - v for v in ys]), [(k - 1 - v, g) for v, g in heads], k


def _gapped(groups, gap=0.6):
    """y per row (first row on top) with `gap` rows between groups and no header rows, plus
    {group: (y_lo, y_hi)} for row_brackets."""
    ys, k, prev = [], 0.0, object()
    for g in groups:
        if ys and g != prev:
            k += gap
        ys.append(k)
        k, prev = k + 1, g
    top = ys[-1] if ys else 0.0
    y = np.array([top - v for v in ys])
    spans = {}
    for g, yi in zip(groups, y):
        lo, hi = spans.get(g, (yi, yi))
        spans[g] = (min(lo, yi), max(hi, yi))
    return y, spans, top + 1


def _heads(ax, heads):
    for yh, g in heads:
        ax.text(-0.02, yh, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.5, fontweight="bold")


def dumbbell(ax, df, group_col=None, note_col=None, brackets=False):
    """One row per test, df order top to bottom. Open dot = comparator (median of donor means),
    filled dot = MF / malignant arm coloured by direction, faint dots = donors, stars at right.
    `brackets=True` spaces groups apart instead of giving each a header row and returns their
    y spans for row_brackets, which the caller draws once the figure is final."""
    rng = np.random.default_rng(0)
    if brackets:
        y, spans, n = _gapped(list(df[group_col]))
        heads = []
    else:
        y, heads, n = _layout(df, group_col)
    for yi, (_, r) in zip(y, df.iterrows()):
        c = DIR_COLOR[r["direction"]]
        for v, col in ((r["vals_b"], HC_COL), (r["vals_a"], c)):
            v = np.asarray(v)
            ax.scatter(v, yi + rng.uniform(-0.16, 0.16, len(v)), s=3, color=col, alpha=0.3,
                       linewidth=0, zorder=2)
        ax.plot([r["med_b"], r["med_a"]], [yi, yi], color=c, linewidth=1.6, zorder=3)
        ax.scatter([r["med_b"]], [yi], s=28, facecolor="white", edgecolor=HC_COL,
                   linewidth=1.0, zorder=4)
        tested = r["direction"] != "not tested"
        ax.scatter([r["med_a"]], [yi], s=32, facecolor=c if tested else "white", edgecolor=c,
                   linewidth=1.0, zorder=5)
        note = r[note_col] if note_col and isinstance(r.get(note_col), str) else ""
        ax.text(1.01, yi, f"{sig(r['q'])}  {note}".rstrip(), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6)
    _heads(ax, heads)
    ax.set_yticks(y)
    ax.set_yticklabels(df["ylabel"], fontsize=6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(0, color="#e0e0e0", linewidth=0.6, zorder=0)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return spans if brackets else None


def _x_annot(ax, i, r):
    ax.text(i, 0.97, sig(r["q"]), transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7)
    if r["test"].startswith("paired") and np.isfinite(r["q"]):
        ax.text(i, 0.89, f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=5.5, color="#555555")


def _gene_ticks(ax, df):
    """Gene names as ticks; under each, % of cells detected (purple malignant, blue reactive)."""
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["gene"], fontsize=6.5, style="italic")
    ax.set_xlim(-0.6, len(df) - 0.4)
    for i, (_, r) in enumerate(df.iterrows()):
        for k, (v, col) in enumerate(((r["det_a"], MALIG), (r["det_b"], REACT))):
            ax.text(i, 0, pct(v), ha="center", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                                          y=-13 - 7 * k, units="points"))
    _clean(ax)


def det_label(ax):
    ax.text(0, 0, "% cells\ndetected", ha="right", va="top", fontsize=5.5, color="#555555",
            transform=offset_copy(ax.transAxes, fig=ax.figure, x=-4, y=-13, units="points"))


def paired(ax, df, violin=False, arm="MF"):
    """Clone vs reactive per gene: one line per biopsy (its two donor means), bar = median.
    Per-cell violins behind when `violin`. Returns the y top the caller should scale to."""
    genes = df["gene"].tolist()
    C = {ct: cell_values(ct, genes, arm=arm) for ct in (NONCLONAL, CLONAL)} if violin else {}
    top = 0.0
    for i, (_, r) in enumerate(df.iterrows()):
        xr, xm = i - 0.2, i + 0.2
        a, b = np.asarray(r["vals_a"]), np.asarray(r["vals_b"])
        if violin:
            for x, ct, col in ((xr, NONCLONAL, REACT), (xm, CLONAL, MALIG)):
                v = C[ct][r["gene"]].to_numpy()
                _violin(ax, v, x, col)
                if (v > 0).sum() >= 20:
                    top = max(top, float(np.quantile(v[v > 0], 0.98)))
        for va, vb in zip(a, b):
            ax.plot([xr, xm], [vb, va], color="#9e9e9e", linewidth=0.4, alpha=0.5, zorder=2)
        for x, v, col in ((xr, b, REACT), (xm, a, MALIG)):
            ax.scatter(np.full(len(v), x), v, s=6, color=col, linewidth=0, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2,
                          zorder=4)
        top = max(top, float(np.nanmax(np.r_[a, b, 0.0])))
        _x_annot(ax, i, r)
    _gene_ticks(ax, df)
    return top


def cell_violins(ax, df, arm="MF"):
    """Per-cell violins, reactive | malignant, donor means as dots. For genes the clone does not
    express the empty panel is the result, so it is drawn rather than dropped."""
    rng = np.random.default_rng(0)
    C = {ct: cell_values(ct, df["gene"].tolist(), arm=arm) for ct in (NONCLONAL, CLONAL)}
    for i, (_, r) in enumerate(df.iterrows()):
        for x, ct, col, v in ((i - 0.2, NONCLONAL, REACT, r["vals_b"]),
                              (i + 0.2, CLONAL, MALIG, r["vals_a"])):
            _violin(ax, C[ct][r["gene"]].to_numpy(), x, col)
            v = np.asarray(v)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=6, color=col,
                       edgecolor="white", linewidth=0.2, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2,
                          zorder=4)
        _x_annot(ax, i, r)
    _gene_ticks(ax, df)


def group_brackets(ax, groups, y=1.03):
    """Label runs of genes above a panel: {name: [genes]} in plotted order."""
    i = 0
    for name, gs in groups.items():
        x0, x1 = i - 0.35, i + len(gs) - 1 + 0.35
        ax.plot([x0, x1], [y, y], transform=ax.get_xaxis_transform(), color="#333333",
                linewidth=0.7, clip_on=False)
        ax.text((x0 + x1) / 2, y + 0.015, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5)
        i += len(gs)


def bracket(ax, x0, x1, y, text, h):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="#333333", linewidth=0.6)
    ax.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=6)


def row_brackets(ax, spans, sub=None, pad=6):
    """Bracket + bold label per row group ({name: (y_lo, y_hi)}) left of the y tick labels;
    `sub` = {name: grey second line}. The x is measured from the drawn tick labels, so call it
    once everything else is on the figure."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x_px = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text())
    x = ax.transAxes.inverted().transform((x_px, 0))[0]
    w = 3 * fig.dpi / 72 / ax.get_window_extent(r).width
    tr = offset_copy(ax.get_yaxis_transform(), fig=fig, x=-pad, units="points")
    for name, (lo, hi) in spans.items():
        ax.plot([x, x - w, x - w, x], [lo - 0.35, lo - 0.35, hi + 0.35, hi + 0.35],
                transform=tr, color="#333333", linewidth=0.7, clip_on=False)
        mid, s = (lo + hi) / 2, (sub or {}).get(name)
        ax.text(x - w, mid, name, ha="right", va="center", fontsize=6.5, fontweight="bold",
                transform=offset_copy(tr, fig=fig, x=-3, y=3.5 if s else 0, units="points"))
        if s:
            ax.text(x - w, mid, s, ha="right", va="center", fontsize=5.5, color="#555555",
                    transform=offset_copy(tr, fig=fig, x=-3, y=-4, units="points"))


def _dot(**k):
    return plt.Line2D([], [], linestyle="", marker="o", markersize=5, **k)


def dir_handles(a="MF", b="healthy"):
    h = [_dot(color=UP, label=f"higher in {a} (q < 0.05)"),
         _dot(color=DOWN, label=f"lower in {a} (q < 0.05)"),
         _dot(color=NS, label="not significant"),
         _dot(markerfacecolor="white", markeredgecolor=NS, label="not testable")]
    return h + ([_dot(markerfacecolor="white", markeredgecolor=HC_COL, label=b)] if b else [])


ARM_HANDLES = [_dot(color=REACT, label=LABELS[NONCLONAL]), _dot(color=MALIG, label=LABELS[CLONAL])]


def caption(fig, text):
    """Wrapped caption under everything already drawn (tick labels, legends included)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight() - 0.015,
             textwrap.fill(text, int(fig.get_figwidth() * 21)), fontsize=5.8, color="#555555",
             ha="left", va="top")


STATS_NOTE = (f"Call: mal_cluster. MF = the {len(CALL_SAMPLES)} MF samples with a call; healthy = atlas HC donors. "
              f"Replicate = donor mean log1p CP10K over >= {cfg.MIN_CELLS} cells. Malignant vs "
              f"reactive CD4: paired Wilcoxon signed-rank on the per-biopsy difference; otherwise "
              f"donor-level Mann-Whitney. BH over the tests in this figure (* < 0.05, ** < 0.01, "
              f"*** < 0.001); n.t. = not testable (< {cfg.MIN_DETECTION:.0%} of cells detected in "
              f"both arms, or < {cfg.MIN_DONORS_PER_ARM} donors per arm). Recomputed here: these "
              f"are not nb28's q-values.")

print(f"call cohort: {len(CALL_SAMPLES)} MF samples, "
      f"{adata.obs.loc[IS_MF_CALL, cfg.DONOR_KEY].nunique()} donors | "
      f"MF cells kept {int(IS_MF_CALL.sum()):,} of {int(IS_MF.sum()):,}")
```

```python  # cell 22
# ============================================================================
# §5  Stage layer — the *_stage companions of FB–FG: early vs advanced MF
# ============================================================================
# Stage = stage_class (early IA-IIA | advanced IIB+); MF samples of unknown stage are left out.
# Within a stage each figure re-runs its own test on that stage's samples: paired signed-rank for
# malignant vs reactive CD4, donor-level MWU against healthy otherwise. Advanced vs early is pooled
# across studies: a donor-level MWU on donor means for a stromal row and, for the clone, on the
# per-biopsy malignant - reactive difference averaged per donor, where study and chemistry cancel
# inside the difference. BH over every test in one figure, advanced-vs-early tests included.
_st_call = SAMPLE2STAGE.reindex(CALL_SAMPLES)
STAGE_SAMPLES = {st: sorted(_st_call.index[_st_call == st]) for st in STAGES}
N_NOSTAGE = int((~_st_call.isin(STAGES)).sum())
_s2study = (pd.Series(adata.obs[cfg.STUDY_KEY].astype(str).to_numpy(), index=_samp)
            .groupby(level=0).first())
STAGE_MIX = {st: _s2study.reindex(STAGE_SAMPLES[st]).value_counts() for st in STAGES}
EVA = "advanced vs early"
GAP_TEST = "MWU (donor, gap vs reactive)"
SHORT_ST = {"early": "early", "advanced": "adv."}


def t_gap_stage(genes, label="gap vs Reactive CD4, advanced vs early"):
    """Advanced vs early on the clone's excess over reactive in the same biopsy: per-biopsy
    difference of the two biopsy means, averaged per donor, donor-level Mann-Whitney."""
    genes = list(dict.fromkeys(genes))
    gap, det = {}, []
    for st in STAGES:
        a, da = donor_means(CLONAL, st, genes, key=cfg.SAMPLE_KEY)
        b, db = donor_means(NONCLONAL, st, genes, key=cfg.SAMPLE_KEY)
        both = a.index.intersection(b.index)
        gap[st] = (a.loc[both] - b.loc[both]).groupby(SAMPLE2DONOR.reindex(both).to_numpy()).mean()
        det += [da, db]
    det = pd.concat(det, axis=1).max(axis=1)
    out = []
    for g in genes:
        x, y = gap["advanced"][g].to_numpy(), gap["early"][g].to_numpy()
        rec = {"label": label, "gene": g, "ct": CLONAL, "arm_a": "advanced", "ct_b": CLONAL,
               "arm_b": "early", "test": GAP_TEST, "n_a": len(x), "n_b": len(y),
               "med_a": np.median(x) if len(x) else np.nan,
               "med_b": np.median(y) if len(y) else np.nan,
               "det_a": det[g], "det_b": det[g], "p": np.nan, "reason": ""}
        rec["delta"] = rec["med_a"] - rec["med_b"]
        rec["sign"] = np.sign(rec["delta"])
        if not det[g] >= cfg.MIN_DETECTION:
            rec["reason"] = f"below {cfg.MIN_DETECTION:.0%} detection in every arm"
        elif min(len(x), len(y)) < cfg.MIN_DONORS_PER_ARM:
            rec["reason"] = f"too few donors ({len(x)}, {len(y)}; need {cfg.MIN_DONORS_PER_ARM})"
        elif np.ptp(np.r_[x, y]) == 0:
            rec["reason"] = "no variance"
        else:
            rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
        rec["vals_a"], rec["vals_b"] = x, y
        out.append(rec)
    return out


def stage_mwu(ct, gene, label=None, **extra):
    """ct's MF arm split by stage: early vs healthy, advanced vs healthy, advanced vs early."""
    item = f"{ct}|{gene}|HC"
    return ([dict(t_mwu(ct, gene, st, "HC", label=label), stage=st, item=item, **extra)
             for st in STAGES]
            + [dict(t_mwu(ct, gene, "advanced", "early", label=label), stage=EVA, item=item,
                    **extra)])


def stage_paired(genes, **extra):
    """Clone vs reactive in the same biopsy, early and advanced biopsies apart, then the gap
    advanced vs early."""
    return ([dict(r, stage=st, item=f"{r['gene']}|paired", **extra) for st in STAGES
             for r in t_paired(genes, arm=st)]
            + [dict(r, stage=EVA, item=f"{r['gene']}|paired", **extra)
               for r in t_gap_stage(genes)])


def stage_split(df):
    """(drawn rows, advanced-vs-early rows). Drawn = one row per item and stage, items in their
    first order, early above advanced; each advanced row carries its item's advanced-vs-early
    q and direction as `eva_q` / `eva_dir` -- the pooled test is written beside the row."""
    eva = df[df["stage"] == EVA].set_index("item")
    order = {k: i for i, k in enumerate(dict.fromkeys(df["item"]))}
    d = df[df["stage"].isin(STAGES)]
    d = (d.assign(_o=d["item"].map(order), _s=d["stage"].map(STAGES.index))
         .sort_values(["_o", "_s"], kind="stable").drop(columns=["_o", "_s"]))
    adv = (d["stage"] == "advanced").to_numpy()
    d["eva_q"] = np.where(adv, eva["q"].reindex(d["item"]).to_numpy(), np.nan)
    d["eva_dir"] = np.where(adv, eva["direction"].reindex(d["item"]).fillna("").to_numpy(), "")
    return d, df[df["stage"] == EVA]


def eva_text(q, direction, sep=" "):
    """'vs early ***↑' on an advanced row (↑ = higher in advanced); '' elsewhere."""
    if not isinstance(direction, str) or not direction:
        return ""
    arrow = {"up": "↑", "down": "↓"}.get(direction, "")
    return f"vs early{sep}{sig(q)}{arrow}"


def eva_color(direction):
    return {"up": STAGE_COL["advanced"], "down": STAGE_COL["early"]}.get(direction, "#555555")


def eva_x_annot(ax, df_eva, y=0.8):
    """Advanced-vs-early gap result above each gene of an advanced-stage paired panel."""
    for i, (_, r) in enumerate(df_eva.iterrows()):
        ax.text(i, y, eva_text(r["q"], r["direction"], sep="\n"),
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5,
                color=eva_color(r["direction"]))


def stage_ticks(ax, stages, axis="y"):
    """Colour the tick labels by stage, in drawing order."""
    labs = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for t, st in zip(labs, stages):
        t.set_color(STAGE_COL.get(st, "#111111"))


def with_stage(df):
    """ylabel + ' · early' / ' · adv.'."""
    return df.assign(ylabel=[f"{y} · {SHORT_ST[s]}" for y, s in zip(df["ylabel"], df["stage"])])


def stage_summary(df):
    """Malignant vs reactive (same biopsy) per stage, then the gap advanced vs early."""
    out = []
    for st in STAGES:
        d = df[(df["stage"] == st) & df["test"].astype(str).str.startswith("paired")]
        up = d.loc[d["direction"] == "up", "gene"].tolist()
        dn = d.loc[d["direction"] == "down", "gene"].tolist()
        n = d["n_pairs"].max() if len(d) else np.nan
        out.append(f"{st} ({f'{n:.0f}' if np.isfinite(n) else '?'} biopsies): higher in "
                   f"malignant {', '.join(up) or 'none'}; higher in reactive "
                   f"{', '.join(dn) or 'none'}")
    ch = df[(df["stage"] == EVA) & (df["test"] == GAP_TEST) & df["direction"].isin(["up", "down"])]
    out.append("malignant - reactive gap, advanced vs early: " + (", ".join(
        f"{g} {'higher' if d_ == 'up' else 'lower'} in advanced"
        for g, d_ in zip(ch["gene"], ch["direction"])) or "no change"))
    return out


STAGE_HANDLES = [_dot(color=STAGE_COL[st], label=f"{st} MF (row label colour)") for st in STAGES]
STAGE_NOTE = (
    "Stage = stage_class (early IA-IIA, advanced IIB+): "
    + "; ".join(f"{st} {len(STAGE_SAMPLES[st])} samples / "
                f"{SAMPLE2DONOR.reindex(STAGE_SAMPLES[st]).nunique()} donors ("
                + ", ".join(f"{s} {n}" for s, n in STAGE_MIX[st].items()) + ")"
                for st in STAGES)
    + f"; {N_NOSTAGE} called MF samples of unknown stage left out. Within a stage: the figure's "
    "own test on that stage's samples. Advanced vs early, pooled across studies: donor-level "
    "Mann-Whitney on donor means, and for malignant vs reactive CD4 on the per-biopsy difference "
    "averaged per donor ('vs early' beside an advanced row; ↑ = higher in advanced). BH over every "
    "test in this figure, both kinds.")

for st in STAGES:
    print(f"{st}: {len(STAGE_SAMPLES[st])} samples, "
          f"{SAMPLE2DONOR.reindex(STAGE_SAMPLES[st]).nunique()} donors | "
          + ", ".join(f"{s} {n}" for s, n in STAGE_MIX[st].items()))
print(f"unknown stage, left out: {N_NOSTAGE} of {len(CALL_SAMPLES)} called MF samples")
```

```python  # cell 23
# ============================================================================
# FB  The clone does not instruct the microenvironment; what it does carry
# ============================================================================
# B1 is the T-cell -> stroma ligands the clone would need to instruct fibroblasts; B2, on the
# same y scale, is what separates clone from reactive in the same biopsy. B1 is drawn, not
# dropped: empty panels are the result.
FB1 = {"T-cell -> stroma ligands": ["CD40LG", "LTA", "IL17A", "IL22", "IL11", "TGFB1"]}
FB2 = {"protease inhibitors": ["TIMP1", "CSTB", "SERPINB1", "SERPINB9"],
       "lysosomal cathepsins": ["CTSA", "CTSD", "CTSH"],
       "granzymes": ["GZMB", "GZMM"]}
_g1 = [g for gs in FB1.values() for g in gs]
_g2 = [g for gs in FB2.values() for g in gs]

FB = bh(t_paired(_g1) + t_paired(_g2))
FB["panel"] = ["B1"] * len(_g1) + ["B2"] * len(_g2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True,
                               gridspec_kw={"width_ratios": [len(_g1), len(_g2)],
                                            "wspace": 0.08})
cell_violins(ax1, FB[FB["panel"] == "B1"])
top = paired(ax2, FB[FB["panel"] == "B2"])
top = max(top, *(float(np.nanmax(np.r_[a, b, 0.0])) for a, b in
                 FB.loc[FB["panel"] == "B1", ["vals_a", "vals_b"]].itertuples(index=False)))
ax1.set_ylim(-0.04 * top, top / 0.82)
ax1.set_ylabel(XLAB, fontsize=7)
ax2.tick_params(labelleft=False)
det_label(ax1)
group_brackets(ax1, FB1)
group_brackets(ax2, FB2)
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="lower right",
           bbox_to_anchor=(0.99, 0.99))
DC._finish(fig, fig28b("FB_what_the_clone_carries"), True)
save_source(FB, "FB_source")
logfig("FB", fig28b("FB_what_the_clone_carries"),
       "T-cell -> stroma ligands the clone lacks vs its inhibitor / lysosomal / granzyme cargo",
       "object, mal_cluster cohort; paired signed-rank on per-biopsy means recomputed, BH within figure")
show_tests(FB, ("panel", "gene", "n_pairs", "med_a", "med_b", "n_pos", "n_neg", "det_a",
                "det_b", "q", "direction"))
```

```python  # cell 24
# ============================================================================
# FB_stage  FB split by stage: early biopsies on top, advanced below
# ============================================================================
FBS = bh(stage_paired(_g1, panel="B1") + stage_paired(_g2, panel="B2"))

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharey=True,
                         gridspec_kw={"width_ratios": [len(_g1), len(_g2)], "wspace": 0.08,
                                      "hspace": 0.6})
top = 0.0
for i, st in enumerate(STAGES):
    ax1, ax2 = axes[i]
    d1 = FBS[(FBS["panel"] == "B1") & (FBS["stage"] == st)]
    d2 = FBS[(FBS["panel"] == "B2") & (FBS["stage"] == st)]
    cell_violins(ax1, d1, arm=st)
    top = max(top, paired(ax2, d2, arm=st),
              *(float(np.nanmax(np.r_[a, b, 0.0]))
                for a, b in d1[["vals_a", "vals_b"]].itertuples(index=False)))
    ax1.set_ylabel(f"{st.upper()} MF\n{XLAB}", fontsize=7, color=STAGE_COL[st])
    ax2.tick_params(labelleft=False)
    det_label(ax1)
    group_brackets(ax1, FB1)
    group_brackets(ax2, FB2)
for ax, panel in zip(axes[1], ("B1", "B2")):
    eva_x_annot(ax, FBS[(FBS["panel"] == panel) & (FBS["stage"] == EVA)])
axes[0][0].set_ylim(-0.04 * top, top / 0.72)
fig.subplots_adjust(top=0.8)
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.99, 0.03))
fig.suptitle("T-cell → stroma ligands, protease inhibitors, lysosomal cathepsins and granzymes "
             "in malignant vs reactive CD4, early and advanced MF", fontsize=9, x=0.0, ha="left",
             y=0.97)
print("\n".join(stage_summary(FBS)))
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FB_stage_what_the_clone_carries"), True)
save_source(FBS, "FB_stage_source")
logfig("FB_stage", fig28b("FB_stage_what_the_clone_carries"),
       "FB per stage: T-cell -> stroma ligands and the clone's cargo, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank per stage + donor MWU on "
       "the gap, BH within figure")
show_tests(FBS, ("panel", "stage", "gene", "n_a", "n_b", "med_a", "med_b", "n_pos", "n_neg",
                 "q", "direction"))
```

```python  # cell 25
# ============================================================================
# FC  LIGHT -> LTbetaR / HVEM -> RELB
# ============================================================================
# Read top to bottom as a chain: the ligand on the clone (1), its receptors on fibroblasts (2),
# the transcription factor they drive (3). One x axis, so the three rows are on one scale.
FC_FIBRO = ["F_apCAF", "F_inflammatory", "F_structural"]
prefetch(FC_FIBRO + ["F_mesenchymal"], ["TNFSF14", "TNFRSF14", "LTBR", "CD40", "RELB"])
FC_ROWS = {
    1: t_paired(["TNFSF14"])
       + [t_mwu(CLONAL, "TNFSF14", "MF", "MF", ct_b=NONCLONAL,
                label="vs Reactive CD4 (across donors)"),
          t_mwu(CLONAL, "TNFSF14", "MF", "HC", label="vs healthy-donor CD4")],
    2: [t_mwu(ct, "TNFRSF14") for ct in FC_FIBRO]
       + [t_mwu("F_structural", "LTBR")]
       + [t_mwu(ct, "CD40") for ct in FC_FIBRO],
    3: [t_mwu(ct, "RELB") for ct in ["F_mesenchymal", "F_inflammatory", "F_structural"]],
}
FC = bh([dict(r, row=k) for k, rs in FC_ROWS.items() for r in rs])
FC["ylabel"] = [r["label"] if r["row"] == 1 else f"{it(r['gene'])} · {LABELS[r['ct']]}"
                for _, r in FC.iterrows()]
# TNFSF14 is detected in few cells: say how few, and how many biopsies voted each way
FC["note"] = [("" if r["row"] != 1 else
               (f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓ · " if r["test"].startswith("paired")
                else "") + f"det {pct(r['det_a'])} | {pct(r['det_b'])}")
              for _, r in FC.iterrows()]
FC_TITLES = ["LIGAND  TNFSF14 (LIGHT) in malignant CD4 (filled) vs comparator (open)",
             "RECEPTORS  fibroblasts, MF vs healthy skin",
             "EFFECTOR  RELB in fibroblasts, MF vs healthy skin"]

nr = FC["row"].value_counts().reindex([1, 2, 3]).to_numpy()
fig = plt.figure(figsize=(7.0, 0.26 * nr.sum() + 2.4))
gs = fig.add_gridspec(3, 1, height_ratios=nr + 0.6, hspace=0.5,
                      left=0.40, right=0.76, top=0.9, bottom=0.12)
axes = []
for k in range(3):
    ax = fig.add_subplot(gs[k], sharex=axes[0] if axes else None)
    dumbbell(ax, FC[FC["row"] == k + 1], note_col="note")
    ax.set_title(FC_TITLES[k], fontsize=7, loc="left")
    if k < 2:
        ax.tick_params(labelbottom=False)
    axes.append(ax)
xmax = max(np.nanmax(np.r_[v, 0.0]) for col in ("vals_a", "vals_b") for v in FC[col])
axes[0].set_xlim(-0.02 * xmax, 1.05 * xmax)
axes[-1].set_xlabel(XLAB, fontsize=7)

# the chain: numbered rows joined by downward arrows in the left margin
cy = [(ax.get_position().y0 + ax.get_position().y1) / 2 for ax in axes]
for k, yc in enumerate(cy):
    fig.text(0.03, yc, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold",
             color="white", bbox=dict(boxstyle="circle,pad=0.35", facecolor="#333333",
                                      edgecolor="none"))
for y0, y1 in zip(cy[:-1], cy[1:]):
    fig.add_artist(FancyArrowPatch((0.03, y0 - 0.04), (0.03, y1 + 0.04),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color="#333333", linewidth=1.0))
fig.legend(handles=dir_handles("malignant / MF", "comparator (reactive or healthy)"),
           fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 0.05))
fig.suptitle("LIGHT → LTβR / HVEM → RELB", fontsize=9.5, x=0.0, ha="left",
             y=0.99)
DC._finish(fig, fig28b("FC_light_ltbr_relb"), True)
save_source(FC, "FC_source")
logfig("FC", fig28b("FC_light_ltbr_relb"), "the LIGHT -> LTbetaR/HVEM -> RELB chain",
       "object, mal_cluster cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FC, ("row", "label", "gene", "n_a", "n_b", "med_a", "med_b", "det_a", "q",
                "direction"))
```

```python  # cell 26
# ============================================================================
# FC_stage  FC split by stage: the LIGHT chain in early and in advanced MF
# ============================================================================
# Every FC row twice, early above advanced (row label colour = stage). Row 1: the clone vs
# reactive in the same biopsy and vs healthy-donor CD4; rows 2-3: fibroblasts vs healthy skin.
# The advanced-vs-early result is written beside each advanced row.
FCS_ROWS = {
    1: stage_paired(["TNFSF14"]) + stage_mwu(CLONAL, "TNFSF14", label="vs healthy-donor CD4"),
    2: ([r for ct in FC_FIBRO for r in stage_mwu(ct, "TNFRSF14")]
        + stage_mwu("F_structural", "LTBR")
        + [r for ct in FC_FIBRO for r in stage_mwu(ct, "CD40")]),
    3: [r for ct in ["F_mesenchymal", "F_inflammatory", "F_structural"]
        for r in stage_mwu(ct, "RELB")],
}
FCS = bh([dict(r, row=k) for k, rs in FCS_ROWS.items() for r in rs])
FCS["ylabel"] = [r["label"] if r["row"] == 1 else f"{it(r['gene'])} · {LABELS[r['ct']]}"
                 for _, r in FCS.iterrows()]
FCD, _ = stage_split(FCS)
FCD = with_stage(FCD)
FCD["note"] = [" · ".join(x for x in (
    f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓" if str(r["test"]).startswith("paired")
    and np.isfinite(r["q"]) else "",
    eva_text(r["eva_q"], r["eva_dir"])) if x) for _, r in FCD.iterrows()]

nr = FCD["row"].value_counts().reindex([1, 2, 3]).to_numpy()
fig = plt.figure(figsize=(7.4, 0.26 * nr.sum() + 2.6))
gs = fig.add_gridspec(3, 1, height_ratios=nr + 0.6, hspace=0.5,
                      left=0.40, right=0.74, top=0.9, bottom=0.12)
axes = []
for k in range(3):
    ax = fig.add_subplot(gs[k], sharex=axes[0] if axes else None)
    d = FCD[FCD["row"] == k + 1]
    dumbbell(ax, d, note_col="note")
    stage_ticks(ax, d["stage"])
    ax.set_title(FC_TITLES[k], fontsize=7, loc="left")
    if k < 2:
        ax.tick_params(labelbottom=False)
    axes.append(ax)
xmax = max(np.nanmax(np.r_[v, 0.0]) for col in ("vals_a", "vals_b") for v in FCD[col])
axes[0].set_xlim(-0.02 * xmax, 1.05 * xmax)
axes[-1].set_xlabel(XLAB, fontsize=7)

cy = [(ax.get_position().y0 + ax.get_position().y1) / 2 for ax in axes]
for k, yc in enumerate(cy):
    fig.text(0.03, yc, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold",
             color="white", bbox=dict(boxstyle="circle,pad=0.35", facecolor="#333333",
                                      edgecolor="none"))
for y0, y1 in zip(cy[:-1], cy[1:]):
    fig.add_artist(FancyArrowPatch((0.03, y0 - 0.04), (0.03, y1 + 0.04),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color="#333333", linewidth=1.0))
fig.legend(handles=dir_handles("malignant / MF", "comparator (reactive or healthy)")
           + STAGE_HANDLES, fontsize=6, frameon=False, ncol=3, loc="upper center",
           bbox_to_anchor=(0.55, 0.05))
_nsig = {st: int(FCD.loc[FCD["stage"] == st, "direction"].isin(["up", "down"]).sum())
         for st in STAGES}
_nev = int(FCS.loc[FCS["stage"] == EVA, "direction"].isin(["up", "down"]).sum())
fig.suptitle("LIGHT → LTβR / HVEM → RELB, early and advanced MF", fontsize=9, x=0.0, ha="left",
             y=0.99)
print(f"{_nsig['early']} rows differ in early, {_nsig['advanced']} in advanced MF; "
      f"{_nev} differ advanced vs early")
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FC_stage_light_ltbr_relb"), True)
save_source(FCS, "FC_stage_source")
logfig("FC_stage", fig28b("FC_stage_light_ltbr_relb"),
       "the LIGHT -> LTbetaR/HVEM -> RELB chain, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank + donor-level MWU per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FCS, ("row", "stage", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 27
# ============================================================================
# FD  Th2 axis
# ============================================================================
# Left to right: the transcription factor is on the clone (D1), the cytokines are below
# detection in both CD4 arms (D2, drawn on D1's scale), the receptor and its output are on the
# stroma (D3). IL4R is shown in all five fibroblast subsets: the reproducibility is the point.
FD1 = ["GATA3", "BATF", "STAT3", "TOX"]
FD2 = ["IL4", "IL5", "IL13", "IL9", "IL31", "IL22"]
FD3 = ([("IL4R", ct, "receptor") for ct in ["F_structural", "F_inflammatory", "F_apCAF",
                                            "F_mesenchymal", "F_reticular"]]
       + [(g, "F_structural", "output") for g in ["HAS2", "POSTN", "TNC"]]
       + [("TSLP", "Mac_FOLR2", "alarmin")])
prefetch(["F_structural", "F_inflammatory", "F_apCAF", "F_mesenchymal", "F_reticular",
          "Mac_FOLR2"], ["IL4R", "HAS2", "POSTN", "TNC", "TSLP"])

FD = bh(t_paired(FD1) + t_paired(FD2) + [dict(t_mwu(ct, g), group=grp) for g, ct, grp in FD3])
FD["panel"] = ["D1"] * len(FD1) + ["D2"] * len(FD2) + ["D3"] * len(FD3)
FD["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FD.iterrows()]

fig = plt.figure(figsize=(12.5, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[len(FD1), len(FD2), 2.3, 4.2], wspace=0.12,
                      left=0.05, right=0.93, top=0.8, bottom=0.2)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[3])
top = paired(ax1, FD[FD["panel"] == "D1"], violin=True)
cell_violins(ax2, FD[FD["panel"] == "D2"])
ax1.set_ylim(-0.04 * top, top / 0.82)
ax1.set_ylabel("log1p CP10K (violin = cells, dots = donor means)", fontsize=6.5)
ax2.tick_params(labelleft=False)
det_label(ax1)
dumbbell(ax3, FD[FD["panel"] == "D3"], group_col="group")
ax3.set_xlabel(XLAB, fontsize=7)
_circ = [g for g in FD1 if g in CALL_PROGRAM]
ax1.set_title("D1  clone: transcription factors (paired)"
              + (f"\n† in the calling program: {', '.join(_circ)}" if _circ else ""),
              fontsize=7.5, loc="left")
ax2.set_title("D2  Th2 cytokines in CD4", fontsize=7.5, loc="left")
ax3.set_title("D3  stroma: receptor and output, MF vs healthy", fontsize=7.5, loc="left")
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.05, 0.02))
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=3,
           loc="upper right", bbox_to_anchor=(0.97, 0.02))
fig.suptitle("Th2 axis", fontsize=9.5, x=0.0, ha="left", y=0.99)
DC._finish(fig, fig28b("FD_th2_axis"), True)
save_source(FD, "FD_source")
logfig("FD", fig28b("FD_th2_axis"), "the Th2 axis: TF on the clone, cytokines, stromal answer",
       "object, mal_cluster cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FD, ("panel", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "det_a", "det_b", "q",
                "direction"))
```

```python  # cell 28
# ============================================================================
# FD_stage  FD split by stage: clone panels per stage (rows), stroma per stage (right)
# ============================================================================
FDS = bh(stage_paired(FD1, panel="D1") + stage_paired(FD2, panel="D2")
         + [r for g, ct, grp in FD3 for r in stage_mwu(ct, g, group=grp, panel="D3")])
FDS["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FDS.iterrows()]

fig = plt.figure(figsize=(12.5, 6.8))
gs = fig.add_gridspec(2, 4, width_ratios=[len(FD1), len(FD2), 2.3, 4.2], wspace=0.12,
                      hspace=0.6, left=0.05, right=0.93, top=0.8, bottom=0.14)
top, row_axes = 0.0, []
for i, st in enumerate(STAGES):
    ax1 = fig.add_subplot(gs[i, 0], sharey=row_axes[0][0] if row_axes else None)
    ax2 = fig.add_subplot(gs[i, 1], sharey=ax1)
    top = max(top, paired(ax1, FDS[(FDS["panel"] == "D1") & (FDS["stage"] == st)],
                          violin=True, arm=st))
    cell_violins(ax2, FDS[(FDS["panel"] == "D2") & (FDS["stage"] == st)], arm=st)
    ax1.set_ylabel(f"{st.upper()} MF\nlog1p CP10K", fontsize=6.5, color=STAGE_COL[st])
    ax2.tick_params(labelleft=False)
    det_label(ax1)
    row_axes.append((ax1, ax2))
row_axes[0][0].set_ylim(-0.04 * top, top / 0.72)
for ax, panel in zip(row_axes[1], ("D1", "D2")):
    eva_x_annot(ax, FDS[(FDS["panel"] == panel) & (FDS["stage"] == EVA)])
row_axes[0][0].set_title("D1  clone: transcription factors (paired)"
                         + (f"\n† in the calling program: {', '.join(_circ)}" if _circ else ""),
                         fontsize=7.5, loc="left")
row_axes[0][1].set_title("D2  Th2 cytokines in CD4", fontsize=7.5, loc="left")

ax3 = fig.add_subplot(gs[:, 3])
D3S, _ = stage_split(FDS[FDS["panel"] == "D3"])
D3S = with_stage(D3S)
D3S["note"] = [eva_text(q, d) for q, d in zip(D3S["eva_q"], D3S["eva_dir"])]
dumbbell(ax3, D3S, group_col="group", note_col="note")
stage_ticks(ax3, D3S["stage"])
ax3.set_xlabel(XLAB, fontsize=7)
ax3.set_title("D3  stroma: receptor and output, MF stage vs healthy", fontsize=7.5, loc="left")

fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.05, 0.03))
fig.legend(handles=dir_handles("MF", "healthy") + STAGE_HANDLES, fontsize=6, frameon=False,
           ncol=3, loc="upper right", bbox_to_anchor=(0.97, 0.03))
_d3 = {st: int((D3S.loc[D3S["stage"] == st, "direction"] == "up").sum()) for st in STAGES}
fig.suptitle("Th2 axis, early and advanced MF", fontsize=9.5, x=0.0, ha="left", y=0.99)
print("\n".join([f"stroma above healthy in {_d3['early']}/{len(FD3)} rows in early, "
                 f"{_d3['advanced']}/{len(FD3)} in advanced MF"]
                + stage_summary(FDS[FDS["panel"].isin(["D1", "D2"])])))
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FD_stage_th2_axis"), True)
save_source(FDS, "FD_stage_source")
logfig("FD_stage", fig28b("FD_stage_th2_axis"), "the Th2 axis, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank + donor-level MWU per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FDS, ("panel", "stage", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 29
# ============================================================================
# FE  Hyaluronan — synthesis, fragmentation, receptor
# ============================================================================
# No curated ligand-receptor edge exists for HA, so the claim rests on who makes it (synthesis),
# who cuts it (fragmentation) and who can bind the fragments (CD44) -- on the stroma in E1 (MF vs
# healthy), on the CD4 arms in E2.
FE1 = ([("HAS2", ct, "synthesis") for ct in ["F_structural", "F_mesenchymal"]]
       + [("HAS1", "F_inflammatory", "synthesis")]
       + [("CEMIP", ct, "fragmentation") for ct in ["F_apCAF", "F_inflammatory", "F_reticular",
                                                    "F_structural"]]
       + [("HYAL2", ct, "fragmentation") for ct in ["F_mesenchymal", "Mac_FOLR2", "Mono_CD14",
                                                    "cDC2"]]
       + [("CD44", ct, "receptor") for ct in ["F_reticular", "F_inflammatory", "F_structural"]])
prefetch(sorted({ct for _, ct, _ in FE1}), ["HAS1", "HAS2", "CEMIP", "HYAL2", "CD44"])
prefetch([CLONAL, NONCLONAL], ["CD44", "HYAL3"])

_e1 = [dict(t_mwu(ct, g), group=grp) for g, ct, grp in FE1]

_e2 = (t_paired(["CD44", "HYAL3"])
       + [t_mwu(CLONAL, "CD44", "MF", "HC", label="Malignant vs healthy-donor CD4"),
          t_mwu(NONCLONAL, "CD44", "MF", "HC", label="Reactive vs healthy-donor CD4")])
FE = bh(_e1 + _e2)
FE["panel"] = ["E1"] * len(_e1) + ["E2"] * len(_e2)
FE["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FE.iterrows()]
E2 = FE[FE["panel"] == "E2"]


def _q(gene, test, ct=CLONAL):
    r = E2[(E2["gene"] == gene) & E2["test"].str.startswith(test) & (E2["ct"] == ct)]
    return float(r["q"].iloc[0]) if len(r) else np.nan


def _bars(ax, gene, arms):
    """Bar = median of donor means, dots = donors. `arms` = [(cell type, 'MF' | 'HC')]."""
    rng = np.random.default_rng(0)
    top = 0.0
    for i, (ct, arm) in enumerate(arms):
        v = donor_means(ct, arm, [gene])[0][gene].to_numpy()
        if not len(v):
            continue
        ax.bar(i, np.median(v), width=0.62, color=ARM_COL[ct], alpha=0.3,
               edgecolor=ARM_COL[ct], linewidth=0.8)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=ARM_COL[ct],
                   edgecolor="white", linewidth=0.2, zorder=3)
        top = max(top, float(v.max()))
    short = {CLONAL: "Malignant\nCD4", NONCLONAL: "Reactive\nCD4", HC_CD4: "Healthy-\ndonor CD4"}
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([short[ct] + (f"\n{arm}" if arm in STAGES else "") for ct, arm in arms],
                       fontsize=6)
    ax.set_title(it(gene), fontsize=8)
    _clean(ax)
    return top


def _stroma_bars(ax, df):
    """One slot per test: healthy bar (hollow) | MF bar (coloured by direction), dots = donors,
    bar = median of donor means, star = MF vs healthy."""
    rng = np.random.default_rng(0)
    top = 0.0
    for i, (_, r) in enumerate(df.iterrows()):
        c = DIR_COLOR[r["direction"]]
        for x, v, fc, ec in ((i - 0.2, r["vals_b"], "white", HC_COL), (i + 0.2, r["vals_a"], c, c)):
            v = np.asarray(v)
            if not len(v):
                continue
            ax.bar(x, np.median(v), width=0.36, color=fc, alpha=0.3 if fc != "white" else 1,
                   edgecolor=ec, linewidth=0.8)
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=ec,
                       edgecolor="white", linewidth=0.2, zorder=3)
            top = max(top, float(v.max()))
        ax.text(i, 0.97, sig(r["q"]), transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["ylabel"], rotation=45, ha="right", fontsize=6)
    ax.set_xlim(-0.6, len(df) - 0.4)
    _clean(ax)
    return top


E1 = FE[FE["panel"] == "E1"]
fig = plt.figure(figsize=(10.5, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[len(E1), 2.4, 1.6], wspace=0.3,
                      left=0.06, right=0.98, top=0.82, bottom=0.34)
ax1, ax2, ax3 = (fig.add_subplot(gs[k]) for k in range(3))
t = _stroma_bars(ax1, E1)
ax1.set_ylim(0, t / 0.85)
ax1.set_ylabel(XLAB, fontsize=7)
group_brackets(ax1, {g: d for g, d in E1.groupby("group", sort=False)["gene"]})
ax1.set_title("E1  stroma and myeloid, MF vs healthy", fontsize=7.5, loc="left", pad=16)

t = _bars(ax2, "CD44", [(CLONAL, "MF"), (NONCLONAL, "MF"), (HC_CD4, "HC")])
for k, (x0, x1, q) in enumerate([(0, 1, _q("CD44", "paired")),
                                 (1, 2, _q("CD44", "MWU", NONCLONAL)),
                                 (0, 2, _q("CD44", "MWU", CLONAL))]):
    bracket(ax2, x0, x1, t * (1.08 + 0.13 * k), sig(q), h=0.03 * t)
ax2.set_ylim(0, t * 1.5)
ax2.set_ylabel(XLAB, fontsize=7)
t = _bars(ax3, "HYAL3", [(CLONAL, "MF"), (NONCLONAL, "MF")])
t = t if t > 0 else 1.0
bracket(ax3, 0, 1, t * 1.08, sig(_q("HYAL3", "paired")), h=0.03 * t)
ax3.set_ylim(0, t * 1.3)
ax2.text(0.0, 1.13, "E2  CD4", fontsize=7.5, transform=ax2.transAxes, ha="left")
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=5,
           loc="upper left", bbox_to_anchor=(0.02, 0.04))
fig.suptitle("Hyaluronan: synthesis, fragmentation and CD44 in the stroma and CD4",
             fontsize=9.5, x=0.0, ha="left", y=0.99)
DC._finish(fig, fig28b("FE_hyaluronan"), True)
save_source(FE, "FE_source")
logfig("FE", fig28b("FE_hyaluronan"), "the hyaluronan axis: synthesis, fragmentation, CD44 in stroma and CD4",
       "object, mal_cluster cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FE, ("panel", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q", "direction"))
```

```python  # cell 30
# ============================================================================
# FE_stage  FE split by stage
# ============================================================================
# E1: each stroma row twice (early, advanced) against healthy, with a bracket for advanced vs
# early. E2: CD4 arms per stage, brackets = the tests drawn (paired within stage, clone advanced
# vs early, clone per stage vs healthy CD4); the gap test is in each panel title.
FES = bh([r for g, ct, grp in FE1 for r in stage_mwu(ct, g, group=grp, panel="E1")]
         + stage_paired(["CD44", "HYAL3"], panel="E2")
         + stage_mwu(CLONAL, "CD44", label="Malignant vs healthy-donor CD4", panel="E2"))
FES["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FES.iterrows()]
E1S, _ = stage_split(FES[FES["panel"] == "E1"])
E1S = with_stage(E1S)
E2S = FES[FES["panel"] == "E2"]


def _qs(gene, stage, kind):
    """q of one E2 test: kind = 'paired' | 'gap' | 'mwu' (the clone's own donor MWU)."""
    t = E2S["test"].astype(str)
    hit = {"paired": t.str.startswith("paired"), "gap": t == GAP_TEST,
           "mwu": t == "MWU (donor)"}[kind]
    r = E2S[(E2S["gene"] == gene) & (E2S["stage"] == stage) & hit]
    return float(r["q"].iloc[0]) if len(r) else np.nan


fig = plt.figure(figsize=(16.5, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[len(E1S), 6.5, 5.2], wspace=0.25,
                      left=0.05, right=0.98, top=0.74, bottom=0.36)
ax1, ax2, ax3 = (fig.add_subplot(gs[k]) for k in range(3))
t = _stroma_bars(ax1, E1S)
for i, (_, r) in enumerate(E1S.iterrows()):
    if r["stage"] == "advanced":
        bracket(ax1, i - 0.8, i + 0.2, t * 1.1, eva_text(r["eva_q"], r["eva_dir"]).replace(
            "vs early ", ""), h=0.02 * t)
ax1.set_ylim(0, t * 1.3)
ax1.set_ylabel(XLAB, fontsize=7)
stage_ticks(ax1, E1S["stage"], axis="x")
group_brackets(ax1, {g: d for g, d in E1S.groupby("group", sort=False)["gene"]}, y=1.04)
ax1.set_title("E1  stroma and myeloid, MF stage vs healthy (bracket: advanced vs early)",
              fontsize=7.5, loc="left", pad=16)

_e2 = [(CLONAL, "early"), (NONCLONAL, "early"), (CLONAL, "advanced"), (NONCLONAL, "advanced")]
t = _bars(ax2, "CD44", _e2 + [(HC_CD4, "HC")])
for k, (x0, x1, q) in enumerate([(0, 1, _qs("CD44", "early", "paired")),
                                 (2, 3, _qs("CD44", "advanced", "paired")),
                                 (0, 2, _qs("CD44", EVA, "mwu")),
                                 (0, 4, _qs("CD44", "early", "mwu")),
                                 (2, 4, _qs("CD44", "advanced", "mwu"))]):
    bracket(ax2, x0, x1, t * (1.08 + 0.13 * k), sig(q), h=0.03 * t)
ax2.set_ylim(0, t * 1.8)
ax2.set_ylabel(XLAB, fontsize=7)
ax2.set_title(f"{it('CD44')}  (gap vs reactive, advanced vs early: "
              f"{sig(_qs('CD44', EVA, 'gap'))})", fontsize=6.5)
t = _bars(ax3, "HYAL3", _e2)
t = t if t > 0 else 1.0
for k, (x0, x1, q) in enumerate([(0, 1, _qs("HYAL3", "early", "paired")),
                                 (2, 3, _qs("HYAL3", "advanced", "paired"))]):
    bracket(ax3, x0, x1, t * 1.08, sig(q), h=0.03 * t)
ax3.set_ylim(0, t * 1.3)
ax3.set_title(f"{it('HYAL3')}  (gap, adv. vs early: {sig(_qs('HYAL3', EVA, 'gap'))})",
              fontsize=6.5)
ax2.text(0.0, 1.16, "E2  CD4 by stage", fontsize=7.5, transform=ax2.transAxes, ha="left")
fig.legend(handles=dir_handles("MF", "healthy") + STAGE_HANDLES, fontsize=6, frameon=False,
           ncol=6, loc="upper left", bbox_to_anchor=(0.02, -0.02))
_up = {st: int((E1S.loc[E1S["stage"] == st, "direction"] == "up").sum()) for st in STAGES}
_n1 = len(FE1)
_ev = int(FES.loc[(FES["panel"] == "E1") & (FES["stage"] == EVA), "direction"]
          .isin(["up", "down"]).sum())
fig.suptitle("Hyaluronan: synthesis, fragmentation and CD44 in the stroma and CD4, early and "
             "advanced MF", fontsize=9.5, x=0.0, ha="left", y=0.99)
print("\n".join([f"stroma above healthy in {_up['early']}/{_n1} rows in early, "
                 f"{_up['advanced']}/{_n1} in advanced MF; {_ev} rows differ advanced vs early"]
                + stage_summary(E2S)))
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FE_stage_hyaluronan"), True)
save_source(FES, "FE_stage_source")
logfig("FE_stage", fig28b("FE_stage_hyaluronan"),
       "the hyaluronan axis in stroma and CD4, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank + donor-level MWU per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FES, ("panel", "stage", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 31
# ============================================================================
# FF  The TGF-β loop runs inside the fibroblast compartment
# ============================================================================
# Left: every stage of the loop -- latent ligand, docking, activation, receptor, output -- in
# fibroblasts, MF vs healthy. Right: the ligands and activators in CD4 on the same x scale,
# clone vs reactive in the same biopsy. Myeloid TGFB1 is up too; it belongs with FB (the clone
# does not instruct), not here.
FF_STAGES = {
    "LATENT LIGAND": [("TGFB1", ["F_inflammatory", "F_apCAF"]),
                      ("TGFB3", ["F_inflammatory", "F_apCAF"])],
    "DOCKING": [("LRRC32", ["F_structural", "F_inflammatory", "F_apCAF"])],
    "ACTIVATION": [("ITGAV", ["F_structural", "F_apCAF"]), ("ITGB8", ["F_structural"]),
                   ("THBS1", ["F_structural", "F_inflammatory"]), ("TGM2", ["F_inflammatory"])],
    "RECEPTOR": [("TGFBR2", ["F_reticular", "F_structural", "F_mesenchymal"])],
    "OUTPUT": [(g, ["F_structural"]) for g in ["TNC", "POSTN", "PRSS23", "ADAMTS2", "FAP",
                                               "LOXL2", "SERPINE1", "LTBP2"]],
}
FF_CD4 = ["TGFB1", "TGFB3", "LRRC32", "ITGB8", "THBS1", "TGM2"]
_ff = [(st, g, ct) for st, gs in FF_STAGES.items() for g, cts in gs for ct in cts]
prefetch(sorted({ct for *_, ct in _ff}), sorted({g for _, g, _ in _ff}))

FF = bh([dict(t_mwu(ct, g), group=st, panel="F1") for st, g, ct in _ff]
        + [dict(r, panel="F2") for r in t_paired(FF_CD4)])
SHORT = {ct: LABELS[ct].replace(" fibroblast", "") for *_, ct in _ff}
FF["ylabel"] = [f"{it(r['gene'])} · {SHORT.get(r['ct'], '')}" for _, r in FF.iterrows()]
FF["note"] = [f"Δ {d:+.2f}" if np.isfinite(d) else "" for d in FF["delta"]]

# stages in the order above; within OUTPUT, largest MF - healthy shift on top
F1 = FF[FF["panel"] == "F1"]
F1 = (F1.assign(_s=F1["group"].map({s: i for i, s in enumerate(FF_STAGES)}),
                _e=np.where(F1["group"] == "OUTPUT", -F1["delta"], 0.0))
      .sort_values(["_s", "_e"], kind="stable").drop(columns=["_s", "_e"]))
F2 = FF[FF["panel"] == "F2"]
FF = pd.concat([F1, F2])

# The caption states these; print whether the data reproduces them rather than assume it.
print("claim checks:")
for st, d in F1.groupby("group", sort=False):
    print(f"  {st:<13} {'ok  ' if (d['direction'] == 'up').any() else 'FAIL'} "
          f"{int((d['direction'] == 'up').sum())}/{len(d)} rows up in MF | "
          f"Δ {d['delta'].min():+.2f} to {d['delta'].max():+.2f}")
_up, _dn = (F2.loc[F2["direction"] == k, "gene"].tolist() for k in ("up", "down"))
print(f"  CD4 flat    {'ok  ' if not _up else 'FAIL'} higher in clone: {_up or 'none'} | "
      f"lower in clone: {_dn or 'none'}")
_lr = F2.set_index("gene").loc["LRRC32"]
print(f"  LRRC32      {'ok  ' if _lr['direction'] == 'down' else 'FAIL'} reactive > clone in "
      f"{_lr['n_neg']:.0f}/{_lr['n_pairs']:.0f} biopsies, {sig(_lr['q'])}")


def cd4_hviolins(ax, df, arm="MF"):
    """Horizontal twin of cell_violins: one row per gene (first on top), malignant above
    reactive. Violin = cells, zeros included; dots = per-biopsy means; % of cells detected under
    each gene. Returns the 98th percentile of expressing cells, for the shared x limit."""
    rng = np.random.default_rng(0)
    C = {ct: cell_values(ct, df["gene"].tolist(), arm=arm) for ct in (NONCLONAL, CLONAL)}
    n, top = len(df), 0.0
    y = np.arange(n)[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        for dy, ct, col, v in ((0.2, CLONAL, MALIG, r["vals_a"]),
                               (-0.2, NONCLONAL, REACT, r["vals_b"])):
            c = C[ct][r["gene"]].to_numpy()
            _violin(ax, c, yi + dy, col, horizontal=True)
            if (c > 0).sum() >= 20:
                top = max(top, float(np.quantile(c[c > 0], 0.98)))
            v = np.asarray(v)
            ax.scatter(v, yi + dy + rng.uniform(-0.06, 0.06, len(v)), s=6, color=col,
                       edgecolor="white", linewidth=0.2, zorder=3)
            if len(v):
                ax.vlines(np.median(v), yi + dy - 0.12, yi + dy + 0.12, color="#111111",
                          linewidth=1.2, zorder=4)
        votes = f"  {r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓" if np.isfinite(r["q"]) else ""
        flip = "  reactive > clone" if r["direction"] == "down" else ""
        ax.text(1.01, yi, f"{sig(r['q'])}{votes}{flip}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6,
                color={"up": UP, "down": DOWN}.get(r["direction"], "#333333"))
        for k, (d, col) in enumerate(((r["det_a"], MALIG), (r["det_b"], REACT))):
            ax.text(0, yi, pct(d), ha="right", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=-3,
                                          y=-5 - 6.5 * k, units="points"))
    ax.set_yticks(y)
    ax.set_yticklabels([it(g) for g in df["gene"]], fontsize=6.5)
    ax.set_ylim(-0.7, n - 0.3)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return top


units = len(F1) + 0.6 * (F1["group"].nunique() - 1)
fig = plt.figure(figsize=(9.0, 0.2 * units + 1.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.85], wspace=0.55,
                      left=0.27, right=0.88, top=0.9, bottom=0.12)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
spans = dumbbell(ax1, F1, group_col="group", note_col="note", brackets=True)
top = cd4_hviolins(ax2, F2)
xmax = max([top] + [float(np.nanmax(np.r_[v, 0.0])) for col in ("vals_a", "vals_b")
                    for v in FF[col]])
ax1.set_xlim(-0.02 * xmax, 1.05 * xmax)
ax1.set_xlabel(XLAB, fontsize=7)
ax2.set_xlabel("log1p CP10K (violin = cells, zeros included; dots = biopsy means)", fontsize=7)
ax1.set_title("Fibroblasts, MF vs healthy skin", fontsize=7.5, loc="left")
ax2.set_title("CD4, malignant vs reactive (same biopsy)\n"
              "under each gene: % cells detected, malignant | reactive", fontsize=7, loc="left")
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=5,
           loc="upper left", bbox_to_anchor=(0.02, 0.05))
fig.legend(handles=ARM_HANDLES, fontsize=6, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.98, 0.05))
fig.suptitle("TGF-β loop genes in fibroblasts (MF vs healthy) and CD4 (malignant vs reactive)",
             fontsize=9.5, x=0.0, ha="left", y=0.99)
_span = F1.groupby("group", sort=False)["delta"].agg(["min", "max"])
row_brackets(ax1, spans, sub={st: f"Δ {r['min']:+.2f} to {r['max']:+.2f}"
                              for st, r in _span.iterrows()})
caption(fig,
        ("The loop is self-contained; no T-cell input is required. " if not _up
         else f"{', '.join(_up)} higher in malignant CD4, so the loop is not T-cell-free. ")
        + "Left: each stage of the loop "
        "in fibroblasts, MF vs healthy skin; output genes ordered by effect, largest on top. "
        "Right: the same ligands and activators in CD4 on the same x scale. The latent ligands "
        "and activators move by small amounts and the receptor and output by large ones (Δ = MF "
        "minus healthy, median of donor means, per row and per stage): the claim is a complete "
        "parts list at the detection limit, not a set of large effects. "
        f"LRRC32 is higher in reactive than malignant CD4 ({sig(_lr['q'])}; detected in "
        f"{pct(_lr['det_b'])} vs {pct(_lr['det_a'])} of cells) -- an internal positive control "
        "that CD4 cells are detected for this gene, so a flat clone is not a sensitivity failure. "
        + STATS_NOTE)
DC._finish(fig, fig28b("FF_tgfb_loop"), True)
save_source(FF, "FF_source")
logfig("FF", fig28b("FF_tgfb_loop"),
       "the TGF-beta loop (latent ligand, docking, activation, receptor, output) in fibroblasts "
       "vs the same genes in CD4",
       "object, mal_cluster cohort; donor-level MWU + paired signed-rank recomputed, BH within figure")
show_tests(FF, ("panel", "group", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "det_a",
                "det_b", "q", "direction"))
```

```python  # cell 32
# ============================================================================
# FF_stage  FF split by stage
# ============================================================================
# Left: every loop row twice (early, advanced) vs healthy, rows in FF's order; right: the CD4
# ligands and activators, malignant vs reactive in early (top) and advanced (bottom) biopsies.
FFS = bh([r for grp, g, ct in _ff for r in stage_mwu(ct, g, group=grp, panel="F1")]
         + stage_paired(FF_CD4, panel="F2"))
FFS["ylabel"] = [f"{it(r['gene'])} · {SHORT.get(r['ct'], '')}" for _, r in FFS.iterrows()]
_order = {f"{r['ct']}|{r['gene']}|HC": i for i, (_, r) in enumerate(F1.iterrows())}
_f1 = FFS[FFS["panel"] == "F1"]
F1S, _ = stage_split(_f1.iloc[np.argsort(_f1["item"].map(_order).to_numpy(), kind="stable")])
F1S = with_stage(F1S)
F1S["note"] = [" · ".join(x for x in (f"Δ {d:+.2f}" if np.isfinite(d) else "",
                                      eva_text(q, e)) if x)
               for d, q, e in zip(F1S["delta"], F1S["eva_q"], F1S["eva_dir"])]
F2S = FFS[FFS["panel"] == "F2"]

print("claim checks by stage:")
for st in STAGES:
    for grp, d in F1S[F1S["stage"] == st].groupby("group", sort=False):
        print(f"  {st:<9} {grp:<13} {'ok  ' if (d['direction'] == 'up').any() else 'FAIL'} "
              f"{int((d['direction'] == 'up').sum())}/{len(d)} rows up in MF | "
              f"Δ {d['delta'].min():+.2f} to {d['delta'].max():+.2f}")
_cd4_up = {st: F2S.loc[(F2S["stage"] == st) & (F2S["direction"] == "up"), "gene"].tolist()
           for st in STAGES}
print(f"  CD4 flat   higher in clone: {_cd4_up}")

units = len(F1S) + 0.6 * (F1S["group"].nunique() - 1)
fig = plt.figure(figsize=(10.5, 0.2 * units + 2.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.85], wspace=0.9, hspace=0.25,
                      left=0.24, right=0.86, top=0.92, bottom=0.08)
ax1 = fig.add_subplot(gs[:, 0])
spans = dumbbell(ax1, F1S, group_col="group", note_col="note", brackets=True)
stage_ticks(ax1, F1S["stage"])
top, axr = 0.0, []
for i, st in enumerate(STAGES):
    ax = fig.add_subplot(gs[i, 1], sharex=ax1)
    top = max(top, cd4_hviolins(ax, F2S[F2S["stage"] == st], arm=st))
    ax.set_title(f"CD4 {st} MF, malignant vs reactive (same biopsy)", fontsize=7, loc="left",
                 color=STAGE_COL[st])
    axr.append(ax)
ev = F2S[F2S["stage"] == EVA]
for yi, (_, r) in zip(np.arange(len(ev))[::-1], ev.iterrows()):
    axr[1].text(1.01, yi - 0.32,
                eva_text(r["q"], r["direction"]).replace("vs early", "gap vs early"),
                transform=axr[1].get_yaxis_transform(), ha="left", va="center", fontsize=5,
                color=eva_color(r["direction"]))
xmax = max([top] + [float(np.nanmax(np.r_[v, 0.0])) for col in ("vals_a", "vals_b")
                    for v in pd.concat([F1S, F2S])[col]])
ax1.set_xlim(-0.02 * xmax, 1.05 * xmax)
ax1.set_xlabel(XLAB, fontsize=7)
axr[1].set_xlabel("log1p CP10K (violin = cells, zeros included; dots = biopsy means)", fontsize=7)
ax1.set_title("Fibroblasts, MF stage vs healthy skin", fontsize=7.5, loc="left")
fig.legend(handles=dir_handles("MF", "healthy") + STAGE_HANDLES, fontsize=6, frameon=False,
           ncol=4, loc="upper left", bbox_to_anchor=(0.02, 0.04))
fig.legend(handles=ARM_HANDLES, fontsize=6, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.98, 0.04))
_ok = {st: sum(bool((d["direction"] == "up").any())
               for _, d in F1S[F1S["stage"] == st].groupby("group", sort=False))
       for st in STAGES}
_ns = F1S["group"].nunique()
fig.suptitle("TGF-β loop genes in fibroblasts and CD4, early and advanced MF", fontsize=9,
             x=0.0, ha="left", y=0.99)
print(f"{_ok['early']}/{_ns} loop stages up in early, {_ok['advanced']}/{_ns} in advanced MF "
      "fibroblasts; " + ("CD4 flat in both" if not any(_cd4_up.values()) else
                         "higher in malignant CD4: " + "; ".join(f"{st} {', '.join(v)}"
                                                                 for st, v in _cd4_up.items() if v)))
row_brackets(ax1, spans)
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FF_stage_tgfb_loop"), True)
save_source(FFS, "FF_stage_source")
logfig("FF_stage", fig28b("FF_stage_tgfb_loop"),
       "the TGF-beta loop in fibroblasts vs the same genes in CD4, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; donor-level MWU + paired signed-rank per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FFS, ("panel", "stage", "group", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 33
# ============================================================================
# FG  Does the stroma track tumour burden? (28_mc §D)
# ============================================================================
# The strongest burden -> stroma association 28_mc §D found: robust rows first (pooled AND within
# study), else the smallest pooled q. One dot per MF sample, coloured by study; burden = the
# malignant share of the sample's mal_cluster-called T cells.
DT = pd.read_csv(cfg.tab("D_burden_stroma"))
BURDEN = pd.read_csv(cfg.tab("burden"), index_col=0)
_t = DT[DT["status"] == "tested"].sort_values(["robust", "q"], ascending=[False, True])
top_ = _t.iloc[0]
row_, readout_ = top_["stroma_row"], top_["readout"]
STROMA_COARSE = {**cfg.FIBRO_COARSE, **cfg.MYELOID_COARSE}
if readout_.startswith("share_of_"):
    _comp = cfg.FIBRO_COARSE if readout_ == "share_of_fibroblast" else cfg.MYELOID_COARSE
    _m = IS_MF & np.isin(LV, [lv for lvs in _comp.values() for lv in lvs])
    _all = pd.Series(_samp[_m]).value_counts()
    _row = pd.Series(_samp[_m & np.isin(LV, STROMA_COARSE[row_])]).value_counts()
    vals = (_row.reindex(_all.index, fill_value=0) / _all)[_all >= cfg.MIN_CELLS]
    ylab = f"{row_} share of {readout_.split('_')[-1]}s"
else:
    _m = IS_MF & np.isin(LV, STROMA_COARSE[row_])
    _g = pd.DataFrame({"s": _samp[_m], "v": adata.obs[readout_].to_numpy()[_m]}).groupby("s")["v"]
    vals = _g.median()[_g.size() >= cfg.MIN_CELLS]
    ylab = f"{readout_} in {row_}\n(sample median)"
pl = BURDEN.join(vals.rename("value"), how="inner")

fig, ax = plt.subplots(figsize=(5, 3))
_cmap = plt.get_cmap("tab20")
for k, s_ in enumerate(sorted(pl[cfg.STUDY_KEY].unique())):
    d_ = pl[pl[cfg.STUDY_KEY] == s_]
    ax.scatter(d_["burden"], d_["value"], s=12, color=_cmap(k % 20), label=s_, linewidth=0)
ax.set_xlabel("tumour burden (malignant share of T cells)", fontsize=7)
ax.set_ylabel(ylab, fontsize=7)
_name = readout_.replace("fam_", "").replace("bal_", "").replace("share_of_", "share of ")
if top_["q"] < 0.05:
    ttl = (f"{_name} in {row_} {'rises' if top_['rho'] > 0 else 'falls'} with tumour burden "
           f"(rho {top_['rho']:+.2f}, q {top_['q']:.1g}; within study rho "
           f"{top_['rho_within_study']:+.2f}{'' if top_['robust'] else ', n.s.'})")
else:
    ttl = (f"No stromal readout tracks tumour burden ({int(_t['n_samples'].max())} MF samples; "
           f"strongest: {_name} in {row_}, q {top_['q']:.2f})")
ax.set_title(f"{_name} in {row_} vs tumour burden, MF samples", fontsize=7, loc="left")
ax.legend(fontsize=5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
_clean(ax)
DC._finish(fig, fig28b("FG_burden_stroma"), True)
pl.to_csv(tab28b("FG_source"))
logfig("FG", fig28b("FG_burden_stroma"), "does the stroma track tumour burden across MF samples",
       "28_mc §D (deg28mc_D_burden_stroma) + per-sample values recomputed from the object")
print(f"FG: {len(_t)} tests, {int(_t['robust'].sum())} robust | drawn: {row_} {readout_} | {ttl}")
```

```python  # cell 34
# ============================================================================
# FG_stage  FG split by stage: does the burden -> stroma association hold within a stage?
# ============================================================================
# FG's readout, dots coloured by stage. Spearman rho within early and within advanced samples;
# the readout advanced vs early is a donor-level Mann-Whitney on donor means (pooled across
# studies). BH over the three tests.
from scipy.stats import spearmanr

PLS = pl.assign(stage=SAMPLE2STAGE.reindex(pl.index).to_numpy())
_n_out = int((~PLS["stage"].isin(STAGES)).sum())
PLS = PLS[PLS["stage"].isin(STAGES)]
_rows = []
for st in STAGES:
    d_ = PLS[PLS["stage"] == st]
    ok = len(d_) >= cfg.MIN_DONORS_PER_ARM and d_["value"].nunique() > 1
    r_, p_ = spearmanr(d_["burden"], d_["value"]) if ok else (np.nan, np.nan)
    _rows.append({"test": f"spearman {st}", "n": len(d_), "rho": r_, "p": p_})
_dv = PLS.groupby("donor").agg(value=("value", "mean"), stage=("stage", "first"))
x_, y_ = (_dv.loc[_dv["stage"] == st, "value"].to_numpy() for st in ("advanced", "early"))
_p = (float(mannwhitneyu(x_, y_, alternative="two-sided").pvalue)
      if min(len(x_), len(y_)) >= cfg.MIN_DONORS_PER_ARM and np.ptp(np.r_[x_, y_]) > 0 else np.nan)
_rows.append({"test": EVA, "n": len(x_) + len(y_), "rho": np.nan, "p": _p,
              "med_advanced": np.median(x_) if len(x_) else np.nan,
              "med_early": np.median(y_) if len(y_) else np.nan})
FGS = pd.DataFrame(_rows)
FGS["q"] = np.nan
_m = FGS["p"].notna()
if _m.any():
    FGS.loc[_m, "q"] = multipletests(FGS.loc[_m, "p"], method="fdr_bh")[1]

fig, ax = plt.subplots(figsize=(5, 3))
for st in STAGES:
    d_ = PLS[PLS["stage"] == st]
    ax.scatter(d_["burden"], d_["value"], s=12, color=STAGE_COL[st], linewidth=0,
               label=f"{st} ({len(d_)} samples)")
ax.set_xlabel("tumour burden (malignant share of T cells)", fontsize=7)
ax.set_ylabel(ylab, fontsize=7)
ax.set_title(f"{_name} in {row_} vs tumour burden, early and advanced MF", fontsize=7,
             loc="left")
ax.legend(fontsize=5.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
_clean(ax)
caption(fig, f"Stage = stage_class (early IA-IIA, advanced IIB+); {_n_out} samples of unknown "
             "stage left out. One dot per MF sample. BH over the two within-stage Spearman tests "
             "and the advanced-vs-early Mann-Whitney (donor means, pooled across studies).")
DC._finish(fig, fig28b("FG_stage_burden_stroma"), True)
PLS.to_csv(tab28b("FG_stage_source"))
FGS.to_csv(tab28b("FG_stage_tests"), index=False)
logfig("FG_stage", fig28b("FG_stage_burden_stroma"),
       "does the stroma track tumour burden within early and within advanced MF",
       "28_mc §D's strongest readout; Spearman per stage + donor MWU advanced vs early, BH")
print(FGS.to_string(index=False, float_format="%.3g"))
```

```python  # cell 35
# ============================================================================
# §R  What was written
# ============================================================================
IDX = pd.DataFrame(FIG_INDEX)
IDX.to_csv(tab28b("figure_index"), index=False)
print(IDX.to_string(index=False))
print(f"\n{len(IDX)} figures -> {cfg.FIG_DIR}/{PREFIX}*.png")
print(f"source tables -> {cfg.TAB_DIR}/{PREFIX}*.csv")
print("nothing under deg28_* was written; nb28's tables are the input, not the output")
```

# Part III — amoeboid motility machinery  (was `28c_amoeboid_motility_mc`)

*Self-contained: a different object (`skin_T_tcr_annotated_v4.h5ad`, backed) and its own `amb28cmc_*` artifacts. It does not read Parts I–II.*

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

```python  # cell 37
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

```python  # cell 38
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

```python  # cell 39
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

```python  # cell 40
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

```python  # cell 41
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

```python  # cell 42
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

```python  # cell 43
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

```python  # cell 44
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

```python  # cell 45
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

```python  # cell 46
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

```python  # cell 47
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
