# old/28_tme_degradome.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28_tme_degradome.ipynb` and is never run._

# 28 — The cutaneous degradome in MF skin (atlas **v2**)

`docs/cutaneous-matrix-summary.md` describes the cutaneous matrix as a set of compartments —
fibrillar and basement-membrane collagens, elastic fibre, proteoglycan and matricellular
scaffolds, the enzyme classes that dismantle each of them, hyaluronan turnover, and the
adhesion machinery a lymphocyte uses to move through what is left. This notebook asks, gene by
gene, how each compartment differs between the disease arm and healthy skin, and which signals
the malignant clone sends into the stroma.

**Three comparisons, one results cell, no figures.**

| § | comparison | design | replicate |
|---|---|---|---|
| **A** | CD4 clonal vs non-clonal | paired **within biopsy** | sample |
| **B** | CD4 clonal / non-clonal / external healthy; myeloid and fibroblast MF / HC / external healthy, at both resolutions | unpaired, per arm pair | donor |
| **C** | curated ligand–receptor edges, CD4 ↔ fibroblast and myeloid | paired within biopsy, plus a specificity null | sample |

§0–§3 are gates and setup and print only what they must. §A–§C compute and write tables and
print nothing but progress. **§R at the bottom is the only place results appear**, and it prints
the comparisons that were run followed by the significant rows only. A non-significant number
appears nowhere except as a count.

## The malignancy call — `mal_tcr` only

`helpers/malignancy.py` assembles three calls. **This analysis uses `mal_tcr` and nothing else.**
It is ALICE TCR clonality: a sequence fact, a CDR3 that recurs in a founder family. `mal_knn`
and `mal_cnv` survive in §2 as QC — the concordance table and, above all, `mal_knn`'s positive
rate in *healthy* CD4, which is its false-positive floor and the reason it is not used here.

- arms are **`clonal`** vs **`non-clonal`**, MF samples only;
- a CD4 cell with no recovered CDR3 is `CD4_unassessed` and is **excluded, not counted as
  benign** — untested is not the same as negative;
- the object on disk was *built* with `mal_knn` as `cfg.MALIG_HEADLINE`, so the notebook derives
  its own `arm_level` column rather than touching the frozen config or the build. Stromal levels
  are identical under either call; only the CD4 rows change.

**The price is n.** The paired design reaches far fewer samples on `mal_tcr` than on `mal_knn`,
because only MF samples that carry TCR contribute and both arms must clear `cfg.MIN_CELLS` (25).
§R prints `n_pairs` next to every p for exactly that reason.

## Unit of analysis

Expression is `X` = log1p CP10K with the size factor summed over all 42,347 genes **before** the
panel subset; §0 gates that and refuses to continue if it drifts. §A and §C take the **sample**
as the replicate and one `clonal − non-clonal` delta *within* each biopsy, so donor, study,
chemistry, biopsy site and treatment are fixed by construction. §B takes the **donor** as the
replicate and is *not* blocked by study: MF vs HC mixes cohorts and the external arm cannot be
blocked at all, so a significant row means the donor distributions separate, not that study
effects are ruled out.

These panels are zero-inflated. `median_delta` is often exactly 0 while the signed-rank is
decisive, so **direction is carried by `pct_up`** (the share of paired samples that moved up),
never by the median. A score with no nonzero paired delta **abstains** — a coverage statement,
not a null result, and §R keeps the two apart.

## Run order

`§0` build gate → `§1` coverage → `§2` malignancy QC and the `mal_tcr` grouping → `§3` external
reference → `§A` → `§B` → `§C` → `§R`. The build is `jobs/run_degradome_build.sh`; the external
object is `jobs/download_external_skin.sh` then `jobs/run_external_degradome_build.sh` (both
bsub, not notebook steps), and both must be re-run whenever `cfg.panel_genes()` changes.

```python  # cell 1
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

import degradome_data as cfg          # config only; frozen
import degradome_helpers as D         # build + gate
import degradome_cells as DC          # the single-cell analysis layer
import malignancy as MAL              # the three calls

cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)

# The one call this analysis uses. cfg.MALIG_HEADLINE stays mal_knn because that is what the
# OBJECT was built with; changing it would invalidate the file on disk.
ARM_CALL = "mal_tcr"
CLONAL, NONCLONAL = cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE
ARM_NAME = {CLONAL: "clonal", NONCLONAL: "non-clonal"}
ARM_LEVEL = "arm_level"

adata = D.load_degradome_adata()
eq = D.cached("equivalence",
              lambda: {"max_abs_diff": float(D.assert_build_equivalence(adata))})
assert eq["max_abs_diff"] < 1e-6, f"normalization gate failed: {eq}"

PANEL = [g for g in cfg.panel_genes() if g in set(map(str, adata.var_names))]
print(f"\nnormalization gate ok: max |lognorm(panel) - lognorm(full)| = "
      f"{eq['max_abs_diff']:.3g}")
print(f"scope {cfg.DISEASES} skin | {adata.n_obs:,} cells, "
      f"{adata.obs[cfg.DONOR_KEY].nunique()} donors, "
      f"{adata.obs[cfg.SAMPLE_KEY].nunique()} samples, "
      f"{adata.obs[cfg.STUDY_KEY].nunique()} studies | panel {len(PANEL)} genes")
print(f"analysis call: {ARM_CALL} (the object was built on {cfg.MALIG_HEADLINE}; "
      f"only the CD4 rows differ)")
```

```
nb22 sidecar reconciled: 20 levels, 328,461 cells, zero drift
skin_degradome.h5ad: 985,273 cells x 503 genes, 112 donors, 123 samples, 12 studies
  scope ['MF', 'HC']: kept 1,002,481 of 1,345,527 skin cells; dropped 17,208 with no identity
  DEGRADOME-ONLY object: var is a hand-curated 503-gene panel. Not for HVG selection, PCA, UMAP, clustering or genome-wide DE. Use skin_degradome_pseudobulk_full.parquet for anything needing all genes.
[cache] equivalence <- equivalence.json

normalization gate ok: max |lognorm(panel) - lognorm(full)| = 0
scope ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies | panel 503 genes
analysis call: mal_tcr (the object was built on mal_knn; only the CD4 rows differ)
```

```python  # cell 2
# ============================================================================
# §1  Coverage gate + the abstention table
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
print(f"\nclaimable: {len(CLAIM)} of {len(cfg.CLAIM_LEVELS)} levels | "
      f"fibroblast {FIBRO} | myeloid {MYELOID}")
print(f"below the donor gate (reported, never quoted): {BELOW_GATE or 'none'}")
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
  CD4_malignant   True   215379            9              18                 38       True                     True                        True
   CD4_reactive   True   110689            9              18                 38       True                     True                        True
 CD4_unassessed  False     2149           12               3                  0       True                    False                       False
       CD4_Treg  False    35086            3              16                 35       True                    False                        True
            CD8  False    43858            8              17                 37       True                     True                        True
              B  False    24810            0              11                 26       True                    False                        True
         Plasma  False    11481            1              15                 24       True                    False                        True
   Keratinocyte  False   212693           21              17                 31       True                     True                        True
       Vascular  False   114799           21              17                 37       True                     True                        True
           Mast  False     3483            6               5                  9       True                     T
... [2564 more chars truncated]
```

```python  # cell 3
# ============================================================================
# §2  Malignancy QC, then the grouping this analysis actually uses
# ============================================================================
mal = MAL.load_malignancy(adata.obs["cell_id"] if "cell_id" in adata.obs
                          else adata.obs_names,
                          has_tcr=adata.obs.get("has_tcr"), verbose=True)
mal.index = adata.obs_names

conc = MAL.concordance(mal, by=adata.obs[cfg.STAGE_KEY].astype(str).to_numpy())
conc.to_csv(cfg.tab("malignancy_concordance"), index=False)

# ---- QC only. This is the whole argument for mal_tcr: mal_tcr finds no clonal cell in healthy
# skin, and whatever mal_knn calls there is error.
IS_CD4 = (adata.obs[cfg.CELLTYPE_SRC].astype(str) == "CD4").to_numpy()
IS_HC_STAGE = (adata.obs[cfg.STAGE_KEY].astype(str) == "HC").to_numpy()
KNN_FP = float(mal.loc[IS_HC_STAGE & IS_CD4 & mal["mal_knn_defined"].to_numpy(),
                       "mal_knn"].mean())
_tcr_hc = mal.loc[IS_HC_STAGE & IS_CD4 & mal["mal_tcr_defined"].to_numpy(), "mal_tcr"]
TCR_FP = float(_tcr_hc.mean()) if len(_tcr_hc) else np.nan
print(f"\nHC CD4 positive rate: mal_knn {KNN_FP:.1%} (its false-positive floor), "
      f"mal_tcr {TCR_FP:.1%} on {len(_tcr_hc):,} CDR3-bearing HC CD4")
print(cfg.FORBIDDEN_CONTRASTS["knn_in_hc"])

# ---- the analysis grouping: stromal levels as built, CD4 re-split on mal_tcr.
# The vocabulary is unchanged (CD4_malignant / CD4_reactive / CD4_unassessed) so every gate and
# ccc_axis.CELL_MAP keep working; only the provenance of the CD4 split changes.
LVL = pd.Series(adata.obs[cfg.GROUPBY].astype(str).to_numpy(), index=adata.obs_names)
LVL = LVL.mask(IS_CD4, pd.Series(MAL.cd4_level(adata.obs[cfg.CELLTYPE_SRC], mal,
                                               call=ARM_CALL).astype(str),
                                 index=adata.obs_names))
adata.obs[ARM_LEVEL] = pd.Categorical(LVL)
LV = LVL.to_numpy()
IS_MF = (adata.obs["disease"].astype(str) == "MF").to_numpy()
IS_HC = (adata.obs["disease"].astype(str) == cfg.HC_DISEASE).to_numpy()

_samp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
TCR_SAMPLES = sorted(set(_samp[IS_MF & IS_CD4 & mal["mal_tcr_defined"].to_numpy()]))
_cd4_mf = IS_MF & IS_CD4
TCR_COHORT = {
    "MF samples with TCR": f"{len(TCR_SAMPLES)} of {len(set(_samp[IS_MF]))}",
    "donors": int(adata.obs.loc[_cd4_mf & np.isin(_samp, TCR_SAMPLES),
                                cfg.DONOR_KEY].nunique()),
    "studies": sorted(adata.obs.loc[IS_MF & np.isin(_samp, TCR_SAMPLES),
                                    cfg.STUDY_KEY].astype(str).unique()),
    "clonal cells": int(((LV == CLONAL) & IS_MF).sum()),
    "non-clonal cells": int(((LV == NONCLONAL) & IS_MF).sum()),
    "excluded, no CDR3": int(((LV == cfg.CD4_UNASSESSED) & IS_MF).sum()),
}
print("\nTCR cohort (the cells sections A and C rest on):")
for k, v in TCR_COHORT.items():
    print(f"  {k}: {v}")

# how the sequence call and the classifier divide the same CD4 cells -- QC, not a result
print("\nmal_tcr level (rows) against the object's mal_knn level (columns), MF CD4:")
print(pd.crosstab(pd.Series(LV[_cd4_mf], name=ARM_CALL),
                  pd.Series(adata.obs[cfg.GROUPBY].astype(str).to_numpy()[_cd4_mf],
                            name=cfg.MALIG_HEADLINE)).to_string())
```

```
has_tcr: 78,442 cells marked TCR-bearing by the V(D)J fold-in cache that atlas obs calls untested (li2024)
         n_defined  frac_defined  n_malignant  frac_malignant_where_defined
mal_tcr   142787.0        0.1449      68843.0                        0.4821
mal_cnv        0.0        0.0000          0.0                           NaN
mal_knn   326068.0        0.3309     215379.0                        0.6605

HC CD4 positive rate: mal_knn 27.4% (its false-positive floor), mal_tcr 0.0% on 858 CDR3-bearing HC CD4
mal_knn calls 23% of healthy CD4 malignant -- it never saw a healthy repertoire. CD4_malignant is not a meaningful level inside the HC arm; in HC, CD4 is reactive by construction.

TCR cohort (the cells sections A and C rest on):
  MF samples with TCR: 38 of 103
  donors: 38
  studies: ['chennareddy2025', 'gaydosik2022', 'herrera2021', 'il4ra2026', 'li2024']
  clonal cells: 68843
  non-clonal cells: 59220
  excluded, no CDR3: 195412

mal_tcr level (rows) against the object's mal_knn level (columns), MF CD4:
mal_knn         CD4_malignant  CD4_reactive  CD4_unassessed
mal_tcr                                                    
CD4_malignant           68843             0               0
CD4_reactive            12696         46524               0
CD4_unassessed         132921         61734             757
```

```python  # cell 4
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


CD4_ARMS = ["clonal", "non-clonal", "external HC"]
TME_ARMS = ["MF", "HC", "external HC"]

cd4_rows = {"CD4": {
    "clonal": (adata, (LV == CLONAL) & IS_MF),
    "non-clonal": (adata, (LV == NONCLONAL) & IS_MF),
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

```python  # cell 5
# ============================================================================
# §A  Intra-sample: CD4 clonal vs non-clonal, paired WITHIN each biopsy, per gene
# ============================================================================
# The replicate is the sample and the statistic is one clonal-minus-non-clonal difference inside
# each biopsy, so donor, study, chemistry, site and treatment are the same tube. No sampling
# imbalance between samples, clone sizes or studies can produce a direction here. The price is
# n_pairs: a sample contributes only if BOTH arms clear cfg.MIN_CELLS.
from statsmodels.stats.multitest import multipletests

A_MASK = IS_MF & np.isin(LV, [CLONAL, NONCLONAL])
cd4_ad = adata[A_MASK].copy()
print(f"A cohort: {int(A_MASK.sum()):,} MF CD4 cells "
      f"({int(((LV == CLONAL) & IS_MF).sum()):,} clonal) in "
      f"{cd4_ad.obs[cfg.SAMPLE_KEY].nunique()} samples, "
      f"{cd4_ad.obs[cfg.DONOR_KEY].nunique()} donors")

# One gene at a time would be 503 passes; all 503 at once puts a 503-column float frame on obs
# for every cell. Chunks of 100 keep both bounded, and BH is applied ONCE at the end over the
# whole panel -- this is one comparison, so it gets one correction.
_parts = []
for _k in range(0, len(PANEL), 100):
    _cols = DC.gene_score_cells(cd4_ad, PANEL[_k:_k + 100], verbose=False)
    _sm = DC.donor_median(cd4_ad, _cols, groupby=ARM_LEVEL, donor_key=cfg.SAMPLE_KEY)
    _parts.append(DC.paired_within(_sm, _cols, pair_key=cfg.SAMPLE_KEY, arm_col=ARM_LEVEL,
                                   ref=NONCLONAL, alt=CLONAL, fdr=False))
    cd4_ad.obs.drop(columns=_cols, inplace=True)

A = pd.concat(_parts, ignore_index=True)
_m = A["p"].notna()
A["q"] = np.nan
if _m.any():
    A.loc[_m, "q"] = multipletests(A.loc[_m, "p"], method="fdr_bh")[1]

A["gene"] = A["score"].str.replace("^gene_", "", regex=True)
A["family"] = A["gene"].map(cfg.GENE_TO_FAMILY).fillna("other")
A["pct_up"] = A["n_pos"] / (A["n_pos"] + A["n_neg"]).replace(0, np.nan)
# Zero-inflated panels: median_delta is often exactly 0 while the signed-rank is decisive, so
# direction is n_pos vs n_neg. A gene with no nonzero paired delta abstains -- that is coverage,
# not a null -- and gets no direction at all.
A["higher"] = np.where(A["n_pos"] > A["n_neg"], "clonal", "non-clonal")
A.loc[A["q"].isna(), "higher"] = ""
A = A.sort_values("q")
A.to_csv(cfg.tab("A_paired_genes"), index=False)

A_NPAIRS = int(A["n_pairs"].max())
print(f"A: {A_NPAIRS} biopsies carry both arms above {cfg.MIN_CELLS} cells | "
      f"{int((A['q'] < 0.05).sum())}/{int(A['p'].notna().sum())} genes at q<0.05, "
      f"{int(A['p'].isna().sum())} abstained -> {cfg.tab('A_paired_genes').name}")
del cd4_ad
```

```
A cohort: 128,063 MF CD4 cells (68,843 clonal) in 38 samples, 38 donors
A: 33 biopsies carry both arms above 25 cells | 43/99 genes at q<0.05, 404 abstained -> deg28_A_paired_genes.csv
```

```python  # cell 6
# ============================================================================
# §B  Cross-arm, unpaired, per gene, at both resolutions
# ============================================================================
# The unit is the DONOR mean, never the cell: at 10^4 cells per arm a cell-level rank test calls
# everything significant and one deep patient can carry it. BH runs over the whole block
# (row x gene x arm pair), which is the honest correction now that there are no figures to
# correct within. Not blocked by study -- MF vs HC mixes cohorts and the external arm is a
# different lab, chemistry and sex.
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
B["arm_pair"] = B["arm_a"] + " vs " + B["arm_b"]
B = B.sort_values(["block", "arm_pair", "q"])
B.to_csv(cfg.tab("B_arm_tests"), index=False)
print(f"\nB: {int((B.status == 'significant').sum())}/{int((B.status != 'not_tested').sum())} "
      f"tested rows at q<0.05, {int((B.status == 'not_tested').sum())} not tested "
      f"-> {cfg.tab('B_arm_tests').name}")
```

```
cd4: 270/503 genes above 1% detection; 159/810 donor-level tests at q < 0.05; never detected: 233; absent from external var: 2
myeloid_low: 410/503 genes above 1% detection; 1026/2861 donor-level tests at q < 0.05; never detected: 93; absent from external var: 2
myeloid_high: 459/503 genes above 1% detection; 1762/6834 donor-level tests at q < 0.05; never detected: 44; absent from external var: 2
fibro_low: 420/503 genes above 1% detection; 1760/3778 donor-level tests at q < 0.05; never detected: 83; absent from external var: 2
fibro_high: 441/503 genes above 1% detection; 2894/7041 donor-level tests at q < 0.05; never detected: 62; absent from external var: 2

B: 7601/21324 tested rows at q<0.05, 12648 not tested -> deg28_B_arm_tests.csv
```

```python  # cell 7
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
CD4 arms re-derived from the supplied level: CD4_malignant 33 samples, CD4_reactive 38 samples | null pools: CD4 side 503 panel genes, stromal side 2000 cube genes
  dropped 14 edge x level keys below 5 paired samples (levels touched: ['Mono_CD16'])
scored 13,018 (sample x arm x edge x level) rows | 394 edge x level keys | 32 samples carry both CD4 arms
paired edge tests: 394 edge x level, 44 at q < 0.05 | median n_pairs = 16
specificity: 7/394 edge x level rows beat both nulls at p < 0.05 (CD4-side pool 503 panel genes -- conservative; stromal-side pool 2000 cube genes)

C: 44/374 edge x level rows at q<0.05, of which 0 also beat both specificity nulls; 20 abstained -> deg28_C_ccc_edges.csv
```

```python  # cell 8
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
P("NB28 — THE CUTANEOUS DEGRADOME IN MYCOSIS FUNGOIDES SKIN (CTCL atlas v2)")
P("SELF-CONTAINED RESULTS REPORT — everything needed to interpret these numbers is below.")
P("#" * 100)

# ---------------------------------------------------------------- 1. the question
P("")
P(RULE)
P("1 · THE QUESTION")
P(RULE)
P("Mycosis fungoides (MF) is a cutaneous T-cell lymphoma in which a malignant CD4 clone sits in")
P("the skin alongside reactive T cells, myeloid cells and fibroblasts. The extracellular matrix")
P("around it is remodelled. This notebook asks three things, cell type by cell type and gene by")
P("gene, over a curated 'degradome' panel -- the proteases that dismantle each matrix")
P("compartment, their inhibitors, the matrix substrates themselves, and the adhesion/motility")
P("machinery a lymphocyte uses to move through what is left:")
P("  (A) inside one biopsy, what does the malignant clone carry that its reactive neighbours do")
P("      not?")
P("  (B) across donors, how do MF cells of each type differ from healthy-control cells of the")
P("      same type, in this atlas and in an independent healthy skin dataset?")
P("  (C) which ligand-receptor edges between the clone and the stroma strengthen in the clone,")
P("      specifically rather than as a by-product of abundance?")

# ---------------------------------------------------------------- 2. the data
P("")
P(RULE)
P("2 · THE DATA")
P(RULE)
P(f"Atlas v2, skin only, disease in {cfg.DISEASES} (MF patients and healthy controls). SS and")
P("other CTCL entities are excluded because malignancy is annotated for MF and not for them.")
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
P(f"This analysis uses ONE call: {ARM_CALL} -- ALICE TCR clonality. A cell is 'clonal' when its")
P("recovered CDR3 belongs to an expanded founder clonotype. That is a sequence fact, not a")
P("classifier output.")
P("  clonal      = malignant CD4, MF samples only")
P("  non-clonal  = CD4 with a recovered CDR3 that is NOT in a founder clonotype, MF samples")
P("  EXCLUDED    = CD4 with no recovered CDR3. Untested is not the same as benign, so these")
P("                cells are dropped rather than counted as reactive.")
P(f"Two other calls exist on this object and NEITHER is used here. mal_knn, a classifier, calls")
P(f"{KNN_FP:.1%} of CD4 cells in HEALTHY skin malignant -- that is its false-positive floor, and")
P(f"it is why it was rejected. mal_tcr calls {TCR_FP:.1%} there. The object on disk was BUILT")
P(f"with mal_knn as its headline, so the notebook re-derives the CD4 split itself; stromal")
P("levels are identical under either call.")
P("TCR cohort that sections A and C rest on: "
  + "; ".join(f"{k} {v}" for k, v in TCR_COHORT.items()))

# ---------------------------------------------------------------- 5. designs
P("")
P(RULE)
P("5 · THE THREE COMPARISONS AND HOW EACH WAS TESTED")
P(RULE)
P("A · INTRA-SAMPLE, PAIRED. Replicate = the BIOPSY. One clonal-minus-non-clonal difference of")
P("    per-sample median expression, computed WITHIN each biopsy, then a two-sided Wilcoxon")
P("    signed-rank over those differences with a sign test alongside. Because both arms come")
P("    from the same tube, donor, study, chemistry, biopsy site, treatment and sequencing batch")
P("    are fixed BY CONSTRUCTION: no sampling imbalance can create a direction. The price is n.")
P(f"    A sample contributes only if both arms clear {cfg.MIN_CELLS} cells. BH over all "
  f"{len(PANEL)} genes (one comparison, one correction).")
P("B · CROSS-ARM, UNPAIRED. Replicate = the DONOR. Each donor's mean expression in an arm is one")
P("    observation; two-sided Mann-Whitney U between arms. Cells within a donor are not")
P("    independent, so there is deliberately NO cell-level p-value anywhere. Each arm needs at")
P(f"    least {cfg.MIN_DONORS_PER_ARM} donors with >= {cfg.MIN_CELLS} cells. BH over the whole")
P("    block (every cell type x gene x arm pair in it). Run at two resolutions:")
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
P("    clonal and the non-clonal CD4 arm on the CD4 side of the edge.")
P(f"    Panel: {len(CA.panel.PRIORITY_PAIRS)} hand-curated edges -> {lr_pairs.edge.nunique()} "
  f"concrete edges x {lr_pairs.stroma_level.nunique()} stromal levels. Restricted to CD4 <-> ")
P("    fibroblast and CD4 <-> myeloid in both directions; nothing else is scored.")
P(f"    An edge is kept only if at least {cfg.MIN_SAMPLES} biopsies carry both CD4 arms for it.")

# ---------------------------------------------------------------- 6. how to read
P("")
P(RULE)
P("6 · HOW TO READ THESE NUMBERS — five rules")
P(RULE)
P("(1) DIRECTION IS pct_up, NOT THE MEDIAN. These panels are zero-inflated: in most biopsies")
P("    both arms are zero for a given gene, so median_delta is exactly 0 even where the")
P("    signed-rank is decisive. pct_up is the share of NONZERO paired differences that went up,")
P("    and 'higher' names the arm. Never read direction off median_delta.")
P("(2) AN ABSTENTION IS NOT A NULL. 'No donor carries 25 cells of this level', 'this gene was")
P("    never detected' and 'the effect was zero' are different findings. Untested rows are")
P("    listed separately at the bottom and are never mixed into the results.")
P("(3) SECTION B IS NOT BLOCKED BY STUDY. MF and HC skin come partly from different cohorts and")
P("    the external arm cannot be blocked at all -- different lab, chemistry, site and sex. A")
P("    significant B row means the donor distributions separate, NOT that cohort and platform")
P("    effects have been ruled out. Section A and C are immune to this by construction.")
P("(4) A HIGH LR SCORE IS NOT SPECIFICITY. sqrt(ligand x receptor) is an abundance product: it")
P("    is large whenever both partners happen to be abundant in the same biopsy. Section C's")
P("    two nulls are what separate a specific edge from an abundant one. There is no")
P("    permutation rank over a full LR resource here, because a 30-pair curated panel cannot")
P("    supply one; these numbers are NOT comparable to LIANA output from other notebooks.")
P("(5) ONLY SIGNIFICANT ROWS (BH q < 0.05) ARE PRINTED BELOW. Everything tested and not")
P("    significant is counted in the inventory and nowhere else. The absence of a gene from a")
P("    table below means it was tested and did not reach q < 0.05, or that it abstained -- the")
P("    inventory and the abstention block distinguish the two.")

# ---------------------------------------------------------------- 7. column glossary
P("")
P(RULE)
P("7 · COLUMN GLOSSARY")
P(RULE)
P("A (paired, per gene)")
P("  delta        median over biopsies of (clonal - non-clonal) per-sample median expression,")
P("               log1p CP10K. Often exactly 0 -- see rule (1).")
P("  pct_up       share of nonzero paired differences that were positive. THE direction column.")
P("  n_pairs      biopsies carrying both arms above the cell gate. n_pos / n_neg split them.")
P("  q            BH-adjusted Wilcoxon signed-rank p over all panel genes. sign_p is the sign")
P("               test on the same differences, reported because at small n it is the more")
P("               trustworthy of the two.")
P("  higher       'clonal' or 'non-clonal'; blank where the gene abstained.")
P("B (unpaired, per gene, per arm pair)")
P("  donor_median_a / _b   median across donors of each donor's mean expression, in arm_a and")
P("               arm_b respectively, log1p CP10K.")
P("  delta        donor_median_b - donor_median_a. Sign follows the column order, so read it")
P("               together with 'higher'.")
P("  n_donors_a / _b       donors that actually entered the test in each arm -- the real n.")
P("  q            BH-adjusted Mann-Whitney p over the whole block.")
P("C (ligand-receptor, paired, with nulls)")
P("  direction    cd4_to_stroma = the clone is the SENDER (its ligand, the stroma's receptor).")
P("               stroma_to_cd4 = the clone is the RECEIVER. These are NOT symmetric: in")
P("               stroma_to_cd4 the stromal ligand is identical between the two arms, because")
P("               it is the same biopsy and the same stromal cells, so the difference is driven")
P("               entirely by the CD4 receptor. Read it as 'does the clone carry more of the")
P("               receptor for what this stroma is sending', never as a claim about stromal")
P("               output.")
P("  edge         LIGAND->RECEPTOR. stroma_level is the non-CD4 partner cell type.")
P("  axis         which of the eight fibrosis axes in section 3 the edge speaks to.")
P("  delta/pct_up/n_pairs/q   as in A, on the edge score instead of a gene.")
P("  spec_p_cd4_side      the CD4-side gene is replaced by panel genes of the SAME ABUNDANCE in")
P("               that arm, and the real edge's mean paired increase is ranked against theirs.")
P("               Small p = the increase is not just 'the whole clonal arm moved'. This pool is")
P("               the curated panel rather than the transcriptome, so it is CONSERVATIVE: it")
P("               can under-call specificity, it cannot manufacture it.")
P("  spec_p_stroma_side   the stromal-side gene is replaced by transcriptome-wide genes of the")
P("               same abundance in that cell type. Small p = the increase is not just 'that")
P("               cell type is deeply sequenced here'.")
P("  spec_p       the LARGER of the two: an edge must beat BOTH nulls. spec_z is how many null")
P("               standard deviations the real increase sits above the null mean.")

# ---------------------------------------------------------------- 8. inventory
def _b_row(tag, label, arms):
    d = B[B.block == tag]
    t = d[d.status != "not_tested"]
    rep = (f"d={int(t[['n_donors_a', 'n_donors_b']].min().min())}-"
           f"{int(t[['n_donors_a', 'n_donors_b']].max().max())}" if len(t) else "-")
    return {"§": "B", "comparison": label, "arms": " / ".join(arms), "replicates": rep,
            "tested": len(t), "significant": int((d.status == "significant").sum()),
            "not tested": int((d.status == "not_tested").sum())}


inv = [{"§": "A", "comparison": "CD4, within biopsy", "arms": "clonal - non-clonal",
        "replicates": f"n={A_NPAIRS}", "tested": int(A["p"].notna().sum()),
        "significant": int((A["q"] < 0.05).sum()),
        "not tested": int(A["p"].isna().sum())}]
inv += [_b_row("cd4", "CD4", CD4_ARMS),
        _b_row("myeloid_low", f"myeloid low res ({len(mye_low)} types)", TME_ARMS),
        _b_row("myeloid_high", f"myeloid high res ({len(mye_high)} types)", TME_ARMS),
        _b_row("fibro_low", f"fibroblast low res ({len(fib_low)} types)", TME_ARMS),
        _b_row("fibro_high", f"fibroblast high res ({len(fib_high)} types)", TME_ARMS)]
inv += [{"§": "C", "comparison": "CD4 <-> fibro/myeloid LR edges",
         "arms": "clonal - non-clonal", "replicates": f"n<={int(C['n_pairs'].max())}",
         "tested": int(C["q"].notna().sum()), "significant": len(C_SIG),
         "not tested": int(C["q"].isna().sum())}]
P("")
P(RULE)
P("8 · EVERY COMPARISON THAT WAS RUN — including the ones that produced nothing")
P(RULE)
P("'replicates' is n biopsies for A and C and the range of donors per arm for B. 'tested' counts")
P("gene x arm-pair rows (B) or genes (A) or edge x cell-type rows (C) that cleared the gates.")
P(pd.DataFrame(inv).to_string(index=False))

# ---------------------------------------------------------------- A
_a = A[A["q"] < 0.05].sort_values("q")
P("")
P(RULE)
P(f"A · CD4 CLONAL vs NON-CLONAL, PAIRED WITHIN BIOPSY — {len(_a)} of "
  f"{int(A['p'].notna().sum())} genes at q<0.05 (n_pairs <= {A_NPAIRS})")
P(RULE)
P("The one contrast in this report that no cohort imbalance, study effect or clone-size")
P("difference can explain: both arms are the same biopsy. A gene here differs between the")
P("malignant clone and the reactive T cells sitting next to it.")
if len(_a):
    P(_a[["gene", "family", "median_delta", "pct_up", "n_pairs", "n_pos", "n_neg", "q",
          "sign_p", "higher"]]
      .rename(columns={"median_delta": "delta"}).round(4).to_string(index=False))
    P("")
    P("  by family: " + "; ".join(f"{k} {v}" for k, v in
                                  _a.groupby("family").size().sort_values(ascending=False)
                                  .items()))
    P("  higher in: " + "; ".join(f"{k} {v}" for k, v in _a["higher"].value_counts().items()))
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
P("blocked by study. Rows flagged 'below donor gate' in the notes are drawn from too few donors")
P("to quote; rows flagged 'ext pooled' are compared against a pooled parent population in the")
P("external reference, not a matched subset.")
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
                               "delta", "n_donors_a", "n_donors_b", "q", "higher"]]
          .round(4).to_string(index=False))

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
P("Only edges that beat both nulls are listed: an increase that survives here is not explained")
P("by the clonal arm moving as a whole, nor by the partner cell type simply being abundant.")
if len(C_SPEC):
    P(C_SPEC.sort_values(["direction", "spec_p", "q"])
      [["direction", "axis", "edge", "stroma_level", "median_delta", "pct_up", "n_pairs",
        "q", "spec_p_cd4_side", "spec_p_stroma_side", "spec_p", "spec_z", "higher"]]
      .rename(columns={"median_delta": "delta"}).round(4).to_string(index=False))
else:
    P("  none.")
P("")
P(f"  {len(C_SIG) - len(C_SPEC)} further edge x cell-type rows rose significantly in the clonal")
P("  arm but did NOT beat both nulls. Those are abundance products, not specific pairs, and are")
P("  deliberately not listed.")
_ctrl = C_SIG[C_SIG.edge.str.contains("S100A8|S100A9") & (C_SIG.higher == "clonal")]
P("  DIRECTION CONTROL — S100A8 and S100A9 are myeloid-restricted genes. If an edge carrying")
P("  them reads higher in the clonal CD4 arm, the cell-type labels are wrong, not the biology")
P("  surprising: "
  + (", ".join(sorted(set(_ctrl.key))) + "  <-- LABELS ARE WRONG" if len(_ctrl) else "clean."))

# ---------------------------------------------------------------- abstentions
P("")
P(RULE)
P("9 · ABSTENTIONS — what this cohort could not be asked, kept apart from what it answered")
P(RULE)
P("None of the following is a negative result. They are places where the data could not support")
P("a test at all, and they must never be reported as 'no difference'.")
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
  f"{len(lr_dropped)} curated panel entries could not be resolved to transcripts at all")
P("   (a shed receptor ectodomain, a rearranged TCR and a zymogen conversion are not gene")
P("   symbols); they are written out rather than dropped silently.")
P("")
P("MACHINE-READABLE TABLES BEHIND THIS REPORT: "
  + ", ".join(cfg.tab(n).name for n in
              ["A_paired_genes", "B_arm_tests", "C_ccc_edges", "ccc_pairs",
               "ccc_pairs_dropped", "coverage", "abstentions", "malignancy_concordance"]))
P("They carry every tested row, significant or not, with the same column names as above.")

txt = cfg.TAB_DIR / f"{cfg.TAB_PREFIX}results.txt"
txt.write_text("\n".join(L) + "\n")
D.append_run_log("NB28 — DEGRADOME, THREE COMPARISONS ON mal_tcr",
                 "\n".join(str(r) for r in inv))
print("\n".join(L))
print(f"\n[full text also written to {txt}]")
```

```
==============================================================================================================
NB28 — DEGRADOME, THREE COMPARISONS ON mal_tcr
==============================================================================================================
{'§': 'A', 'comparison': 'CD4, within biopsy', 'arms': 'clonal - non-clonal', 'replicates': 'n=33', 'tested': 99, 'significant': 43, 'not tested': 404}
{'§': 'B', 'comparison': 'CD4', 'arms': 'clonal / non-clonal / external HC', 'replicates': 'd=5-38', 'tested': 810, 'significant': 159, 'not tested': 0}
{'§': 'B', 'comparison': 'myeloid low res (4 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-84', 'tested': 2861, 'significant': 1026, 'not tested': 2059}
{'§': 'B', 'comparison': 'myeloid high res (12 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-67', 'tested': 6834, 'significant': 1762, 'not tested': 9690}
{'§': 'B', 'comparison': 'fibroblast low res (3 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-59', 'tested': 3778, 'significant': 1760, 'not tested': 2}
{'§': 'B', 'comparison': 'fibroblast high res (6 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-51', 'tested': 7041, 'significant': 2894, 'not tested': 897}
{'§': 'C', 'comparison': 'CD4 <-> fibro/myeloid LR edges', 'arms': 'clonal - non-clonal', 'replicates': 'n<=24', 'tested': 374, 'significant': 44, 'not tested': 20}
####################################################################################################
NB28 — THE CUTANEOUS DEGRADOME IN MYCOSIS FUNGOIDES SKIN (CTCL atlas v2)
SELF-CONTAINED RESULTS REPORT — everything needed to interpret these numbers is below.
####################################################################################################

====================================================================================================
1 · THE QUESTION
====================================================================================================
Mycosis fungoides (MF) is a cutaneous T-cell lymphoma in which a malignant CD4 clone sits in
the skin alongside reactive T cells, myeloid cells and fibroblasts. The extracellular matrix
around it is remodelled. This notebook asks three things, cell type by cell type and gene by
gene, over a curated 'degradome' panel -- the proteases that dismantle each matrix
compartment, their inhibitors, the matrix substrates themselves, and the adhesion/motility
machinery a lymphocyte uses to move through what is left:
  (A) inside one biopsy, what does the malignant clone carry that its reactive neighbours do
      not?
  (B) across donors, how do MF cells of each type differ from healthy-control cells of the
      same type, in this atlas and in an independent healthy skin dataset?
  (C) which ligand-receptor edges between the clone and the stroma strengthen in the clone,
      specifically rather than as a by-product of abundance?

====================================================================================================
2 · THE DATA
====================================================================================================
Atlas v2, skin only, disease in ['MF', 'HC'] (MF patients and healthy controls). SS and
other CTCL entities are excluded because malignancy is annotated for MF and not for them.
  985,273 cells | 112 donors | 123 samples | 12 studies
EXPRESSION UNITS: log1p(CP10K). The library size is summed over all 42,347 genes BEFORE the
  panel columns are subset, so values are comparable across cell types and sequencing
  depths. This is gated, not assumed: max |lognorm(panel) - lognorm(full)| = 0.
EXTERNAL HEALTHY REFERENCE: reynolds2021 (E-MTAB-8142), 188,698 healthy adult skin cells from
  5 donors, a different lab, chemistry and sex mix,
  normalised on the same contract. Its cell labels are coarser than ours, so a level with
  no exact match is compared with its nearest parent POOLED -- that is a reference level,
  not a matched subset. Pooled lev
... [968384 more chars truncated]
```
