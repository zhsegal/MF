# old/27_ccc_subtype.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/27_ccc_subtype.ipynb` and is never run._

# 27 · CCC — the malignant CD4 ligand–receptor map at sub-level resolution

One notebook. Supersedes `old/ccc_v1_01_build_object`, `ccc_v1_02/03` (v1, 13 pooled levels) and
`ccc_v1_04/05` (sub-level re-run). Config is `ccc_data.py` + `ccc_data_sub.py`; every function is in
`ccc_utils.py`. Nothing is defined here.

## What changed, and why every number moved

| | v1 / `ccc_v1_04/05` | here |
|---|---|---|
| ranking | `magnitude_rank`, gated on `cellphone_pvals` | **`specificity_rank`** only |
| runs | ~14 separate `rank_aggregate` calls | **one** run per window |
| order | `ccc_v1_04` subsampled then windowed | label → **window → subsample** |
| n per level | 8,000 cap, levels differ 10× | **common n**, every level equal |
| comparator | `rank_delta` / `malignant_only` | **paired donor contrast** (§3) |
| evidence | cell-unit permutation p | **donor-level** (§2 §3 §4) |
| landing | `gap_rank_ratio < 3`, no null | **within-donor, power-equalised** (§4) |
| ambient | `LR_CAVEATS` blacklist | **regression on donor malignant fraction** (§5) |

Every LIANA magnitude score is monotone in absolute expression, and the permutation null has
SD σ/√n_c — so the p measured cells per level, not evidence. `liana_pipe` also derives
`groupby_subset` from `groupby_pairs` and *physically subsets the object*, so scores from two
different grids were never on a common scale and must not be subtracted. Both are fixed by
construction here: `run_spec_consensus` deletes `magnitude_rank` and `cellphone_pvals` and
asserts they are gone, and there is exactly one grid.

## Roster

18 levels clear the claim gate (≥5 donors at ≥25 cells, CTCL window) — there is no
`report_only` tier. **pDC (1,111 cells) is dropped from the ranked run**: it alone would set the
common n at 1,111 for all 18. It still appears in the coverage table and the forced panel, so
its absence is on the record. The ranked roster is 17 levels at a common n.

Dropped from the roster upstream (`ccc_data_sub.DROP_LEVELS`): `CD4_unassessed`, `Tregs`,
`Plasma`, `Vascular`, `Mast`, `Melanocyte`. That is a scope narrowing, not a refutation — the
`ccc_v1_02/03` tables for them stand.

## Sections

§0 input, labels, roster, claim gate · §1 the single run · §2 donor reproducibility ·
§3 malignant vs reactive, paired · §4 edge landing · §5 controls · §6 forced panel ·
§7 reportable set + figures.

Every section writes `tables/ccc40_*.csv` and appends a self-describing block to
`tables/ccc40_run_log.md` that is readable pasted into a chat with no access to this notebook.

**Kernel: `mrvi_env`** (`neural_nmf_env` has no liana).

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


NB_DIR = next(c.resolve()
              for b in [Path.cwd(), *Path.cwd().parents]
              for c in (b, b / "MF")
              if c.name == "MF" and (c / "data").exists())
sys.path.insert(0, str(NB_DIR / "helpers"))
import ccc_data_sub as cfg     # config only; frozen
import ccc_utils as U          # every function this notebook calls

sc.settings.verbosity = 1
U.set_style()
U.TAB_DIR.mkdir(parents=True, exist_ok=True)
U.FIG_DIR.mkdir(parents=True, exist_ok=True)
U.reset_run_log()

print("NB_DIR", NB_DIR)
print("roster (config):", cfg.KEEP_LEVELS)
print("dropped from the ranked run:", U.DROP_FROM_RUN)
print("\n" + cfg.CAVEAT_BLOCK)
```

```
NB_DIR /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
roster (config): ['CD4_malignant', 'CD4_reactive', 'CD8', 'B', 'Keratinocyte', 'LC', 'cDC', 'DC_LAMP3', 'pDC', 'Mono_moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'F_papillary', 'F_reticular', 'F_mesenchymal', 'F_inflammatory', 'F_apCAF']
dropped from the ranked run: ['pDC']

LIANA permutation p-values are computed with the CELL as the unit and are pseudoreplicated across donors (76,349 malignant CD4 cells come from 29 donors). They size the dots and mark the heatmap cells; they are not evidence. Donor-level inference is the deferred differential phase. Grey heatmap cells mean the pair failed min_cells in that stratum -- read them against the coverage table printed beneath each figure, not as biology. One caveat is specific to this run: splitting a level into k sub-levels multiplies the tested grid by k while each sub-level carries fewer cells and fewer donors, so a pair that was solid on pooled Myeloid can look weaker on every sub-level without any biology having changed. Read the sub-level result against the v1 pooled result (tables/ccc_liana_*.csv), not on its own.
```

## §0 · Input, labels, roster, claim gate

`ccc_skin.h5ad` is not rebuilt — it already holds every myeloid and fibroblast cell, and only
the grouping column and the roster change. What has to be re-established every run: the object
is what the build claims, the sub-label reconciles cell-for-cell against the `22_myeloid_fibro_reannotation` sidecar, and
the five untouched levels (`CD4_malignant`, `CD4_reactive`, `CD8`, `B`, `Keratinocyte`) carry
over unchanged — that last one is what makes the B-cell axis a regression test.

```python  # cell 3
# ---------------------------------------------------------------- object + sub-level label
adata = U.load_ccc_adata()
U.assert_ccc_invariants(adata)

obs = adata.obs.copy(); obs["cell_id"] = obs.index.astype(str)
adata.obs[cfg.GROUPBY] = U.build_ccc_celltype_sub(obs)
recon = U.assert_roster(adata)

roster = adata[adata.obs[cfg.GROUPBY].notna()].copy()
roster.obs[cfg.GROUPBY] = roster.obs[cfg.GROUPBY].cat.remove_unused_categories()
roster.layers[cfg.LAYER] = roster.X
del adata
resource, res_cov = U.load_resource(var_names=roster.var_names)
print(f"\nroster object: {roster.n_obs:,} cells x {roster.n_vars:,} genes, "
      f"{len(roster.obs[cfg.GROUPBY].cat.categories)} levels")
display(recon)
```

```
727,156 cells x 1,832 resource genes
invariants OK: (727156, 1832)
59,658 CD4 negative for mal_tcr_alice with no assessed_tcr_nb30 -> CD4_unassessed (untestable, not benign)
dropping 265,525 cells (36.5%) outside the roster
ccc_celltype_sub
Keratinocyte      88692
CD4_malignant     76349
CD4_reactive      72207
CD8               51831
cDC               30650
F_reticular       23128
B                 19388
DC_LAMP3          19068
F_inflammatory    14144
Mac_FOLR2         13943
Mac_infl          12731
F_apCAF           10236
F_papillary        8262
Mono_moDC          7047
F_mesenchymal      6146
Mac_SPP1_TREM2     3835
LC                 2861
pDC                1113
consensus: 4624 interactions, 1877 subunit genes

roster object: 461,631 cells x 1,832 genes, 18 levels

           level  n_pooled  n_sub_labelled  n_dropped_unk_prolif
0        Myeloid     93435           91248                  2187
1     Fibroblast     66760           61916                  4844
2            CD8     51831           51831                     0
3              B     19388           19388                     0
4   Keratinocyte     88692           88692                     0
5  CD4_malignant     76349           76349                     0
6   CD4_reactive     72207           72207                     0
```

```python  # cell 4
# ---------------------------------------------------------------- the equivalence gate (from nb34)
# BLOCKING. The build kept only the 1,832 resource genes, which is lossless for liana ONLY if
# the lognorm size factor was summed over all 40,821 genes BEFORE the columns were subset.
# Doing it the other way round rescales every cell by ~6.7x and produces plausible, wrong
# scores. Run the same grid on the 3-donor FULL-GENE test object and on the same cells inside
# ccc_skin.h5ad; every statistic must match.
test = sc.read_h5ad(cfg.TESTSET_H5AD)
test.obs[cfg.GROUPBY] = U.build_ccc_celltype_sub(
    test.obs.assign(cell_id=test.obs.index.astype(str)), verbose=False)
test = test[test.obs[cfg.GROUPBY].notna()].copy()
test.layers[cfg.LAYER] = test.X

gate_sub = roster[roster.obs_names.isin(test.obs_names)].copy()
test = test[gate_sub.obs_names].copy()
gate_sub.layers[cfg.LAYER] = gate_sub.X
gate_lv = [lv for lv, n in gate_sub.obs[cfg.GROUPBY].value_counts().items() if n >= cfg.MIN_CELLS][:4]
assert len(gate_lv) >= 2, f"the equivalence testset carries too few usable levels: {gate_lv}"
gate_pairs = U.build_groupby_pairs(gate_lv)

g_full = U.run_spec_consensus(test, resource, gate_pairs, key_added="gate_full")
g_sub = U.run_spec_consensus(gate_sub, resource, gate_pairs, key_added="gate_sub")
a = g_full.set_index(U.KEY_COLS).sort_index()
b = g_sub.set_index(U.KEY_COLS).sort_index()
assert a.index.equals(b.index), "row sets differ between the full-gene and subset objects"
for col in [U.RANK_COL, U.ABUNDANCE_COL, "spec_weight", "scaled_weight", "lr_logfc"]:
    d = float(np.nanmax(np.abs(a[col].to_numpy() - b[col].to_numpy())))
    print(f"  {col:<18} max abs diff {d:.3e}")
    assert d < 1e-6, f"{col} differs -- the lognorm size factor is wrong"
print(f"\nEQUIVALENCE GATE PASSED on {gate_sub.n_obs:,} cells, levels {gate_lv}")
del test, gate_sub, g_full, g_sub
```

```
gate_full: 1,522 edges scored at expr_prop=0.1 over 4 sender levels, 12 ordered level pairs
gate_sub: 1,522 edges scored at expr_prop=0.1 over 4 sender levels, 12 ordered level pairs
  specificity_rank   max abs diff 0.000e+00
  expr_prod          max abs diff 0.000e+00
  spec_weight        max abs diff 0.000e+00
  scaled_weight      max abs diff 0.000e+00
  lr_logfc           max abs diff 0.000e+00

EQUIVALENCE GATE PASSED on 44,122 cells, levels ['CD4_malignant', 'CD4_reactive', 'Keratinocyte', 'CD8']
```

```python  # cell 5
# ---------------------------------------------------------------- window, coverage, claim gate
# Claimability is computed INSIDE the window the runs use. Computed on the whole object it
# would count HC donors no CTCL run ever sees.
ctcl = U.focal_window(roster, disease=cfg.CTCL_DISEASES)
gate = U.claim_gate(ctcl, levels=cfg.KEEP_LEVELS)
CLAIMABLE = [lv for lv in cfg.KEEP_LEVELS if lv in gate.index and gate.loc[lv, "claimable"]]
MAIN_ROSTER = [lv for lv in CLAIMABLE if lv not in U.DROP_FROM_RUN]
gate.to_csv(U.tab("coverage"))
display(gate)

hc = U.focal_window(roster, disease=["HC"], verbose=False)
n_hc_mal = int((hc.obs[cfg.GROUPBY].astype(str) == cfg.CD4_MALIGNANT).sum())
assert n_hc_mal == 0, f"{n_hc_mal} malignant CD4 in healthy skin -- the label is wrong"
hc_gate = U.claim_gate(hc)
HC_LEVELS = [lv for lv in cfg.KEEP_LEVELS if lv in hc_gate.index and hc_gate.loc[lv, "claimable"]]

U.printout(
    "SECTION 0 -- INPUT, LABELS, ROSTER, CLAIM GATE",
    lines=[
        f"object          : {roster.n_obs:,} cells x {roster.n_vars:,} resource genes",
        f"resource        : {cfg.RESOURCE_NAME}, {res_cov['n_interactions']} interactions, "
        f"{res_cov['n_interactions_covered']} fully covered by the data "
        f"({res_cov['frac_interactions_covered']:.1%})",
        f"CTCL window     : {ctcl.n_obs:,} cells, {ctcl.obs[cfg.DONOR_KEY].nunique()} donors, "
        f"{ctcl.obs[cfg.STUDY_KEY].nunique()} studies",
        f"HC window       : {hc.n_obs:,} cells, {hc.obs[cfg.DONOR_KEY].nunique()} donors, "
        f"malignant CD4 = {n_hc_mal} (asserted 0)",
        f"claim gate      : >= {cfg.MIN_SAMPLES} donors at >= {cfg.MIN_CELLS} cells",
        f"claimable       : {len(CLAIMABLE)}/{len(cfg.KEEP_LEVELS)} -> {CLAIMABLE}",
        f"dropped from run: {U.DROP_FROM_RUN} (would set the common n for every level)",
        f"ranked roster   : {len(MAIN_ROSTER)} levels",
        f"HC claimable    : {HC_LEVELS}",
        "study dominance is reported per level and TRAVELS WITH THE ROW; it never gates.",
        "equivalence gate: PASSED (full-gene vs resource-subset agree to 1e-6).",
    ],
    tables={"pooled -> sub reconciliation": recon,
            "coverage / claim gate (CTCL window)": gate.reset_index().rename(
                columns={"index": "level"})},
)
```

```
window: 447,358 cells, 73 donors, 7 studies

                  n_cells  n_donors_present  n_donors_ge25  n_studies  \
ccc_celltype_sub                                                        
Keratinocyte        83919                68             54          7   
CD4_malignant       76349                31             29          6   
CD4_reactive        71488                33             32          6   
CD8                 51531                72             63          7   
cDC                 29952                73             65          7   
B                   19377                65             41          7   
DC_LAMP3            19061                71             45          7   
F_reticular         18522                69             42          7   
F_inflammatory      14131                51             11          6   
Mac_FOLR2           13543                73             55          7   
Mac_infl            12720                66             31          7   
F_apCAF             10043                69             42          7   
Mono_moDC            7017                72             43          7   
F_papillary          7006                71             36          7   
F_mesenchymal        4933                60             34          7   
Mac_SPP1_TREM2       3809                67             21          7   
LC                   2846                69             15          7   
pDC                  1111                64             13          7   

                        top_study  top_study_frac  claimable  study_dominated  
ccc_celltype_sub                                                               
Keratinocyte               li2024           0.491       True            False  
CD4_malignant              li2024           0.570       True            False  
CD4_reactive               li2024           0.747       True            False  
CD8                        li2024           0.689       True            False  
cDC                        li2024           0.427       True            False  
B                          li2024           0.742       True            False  
DC_LAMP3                   li2024           0.882       True             True  
F_reticular       chennareddy2025           0.311       True            False  
F_inflammatory             li2024           0.930       True             True  
Mac_FOLR2         chennareddy2025           0.481       True            False  
Mac_infl                   li2024           0.617       True            False  
F_apCAF           chennareddy2025           0.474       True            False  
Mono_moDC         chennareddy2025           0.547       True            False  
F_papillary                li2024           0.346       True            False  
F_mesenchymal              li2024           0.529       True            False  
Mac_SPP1_TREM2             li2024           0.783       True            False  
LC                         li2024           0.749       True            False  
pDC                        li2024           0.608       True            False  
==============================================================================================================
SECTION 0 -- INPUT, LABELS, ROSTER, CLAIM GATE
==============================================================================================================
object          : 461,631 cells x 1,832 resource genes
resource        : consensus, 4624 interactions, 4484 fully covered by the data (97.0%)
CTCL window     : 447,358 cells, 73 donors, 7 studies
HC window       : 14,273 cells, 9 donors, malignant CD4 = 0 (asserted 0)
claim gate      : >= 5 donors at >= 25 cells
claimable       : 18/18 -> ['CD4_malignant', 'CD4_reactive', 'CD8', 'B', 'Keratinocyte', 'LC', 'cDC', 'DC_LAMP3', 'pDC', 'Mono_moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'F_papillary', 'F_reticular', 'F_mesenchymal', 'F_inflammatory', 'F_apCAF']
dropped from run: ['pDC'] (would set the common n for ever
... [3338 more chars truncated]
```

## §1 · The single run

`SPEC_CONSENSUS` = `AggregateClass(aggregate_meta, [natmi, connectome, logfc])`. All three
member methods carry `permute=False`, so **no permutation is ever computed** — the run is
permutation-free. `n_perms` is passed as an int only because `liana_pipe` overwrites
`consensus_opts` with `'Magnitude'` when it is `None`, and `consensus_opts=["Specificity"]`
raises `KeyError` in liana 1.8.1. `magnitude_rank`, `cellphone_pvals` and `lr_means` are
deleted from the frame and their absence asserted.

**Order matters and was wrong before**: label → `focal_window(CTCL)` → `subsample_common_n`.
Subsampling first spends the cap on cells no run sees, and spends it unevenly — the structural
levels carry HC cells, `CD4_malignant` carries none.

One grid over the whole ranked roster, both directions. Every later section slices this frame.

```python  # cell 7
run_ad, sub_report = U.subsample_common_n(ctcl, MAIN_ROSTER)
pairs = U.build_groupby_pairs(MAIN_ROSTER)
res = U.run_spec_consensus(run_ad, resource, pairs, key_added="spec_ctcl")

# The HC negative control only exists if healthy skin carries two claimable levels. Where it
# does not, that failure IS the result -- the sub-level resolution is a CTCL-window statement.
if len(HC_LEVELS) >= 2:
    hc_ad, hc_report = U.subsample_common_n(hc, HC_LEVELS, verbose=False)
    res_hc = U.run_spec_consensus(hc_ad, resource, U.build_groupby_pairs(HC_LEVELS),
                                  key_added="spec_hc")
    hc_note = (f"{len(HC_LEVELS)} levels, {hc_ad.n_obs:,} cells, no CD4 level exists in HC skin")
else:
    res_hc = pd.DataFrame(columns=U.KEY_COLS + [U.RANK_COL])
    hc_note = (f"NOT RUN -- only {len(HC_LEVELS)} level(s) clear the claim gate in healthy "
               "skin, which is itself the result: this resolution is a CTCL-window statement")

res.to_parquet(U.TAB_DIR / "ccc40_run.parquet", index=False)
res.head(5000).to_csv(U.tab("run_top5000"), index=False)
res_hc.head(5000).to_csv(U.tab("run_hc_top5000"), index=False)
sub_report.to_csv(U.tab("subsample"), index=False)

top = U.top_edges(res, 15)[U.KEY_COLS + [U.RANK_COL, U.ABUNDANCE_COL]]
U.printout(
    "SECTION 1 -- ONE RUN, PERMUTATION-FREE SPECIFICITY CONSENSUS",
    lines=[
        f"methods         : NATMI + Connectome + log2FC, RRA aggregate, rank = {U.RANK_COL}",
        f"grid            : {len(MAIN_ROSTER)} levels, {len(pairs)} ordered level pairs, "
        "both directions, ONE run",
        f"window          : CTCL ({', '.join(cfg.CTCL_DISEASES)}) applied BEFORE subsampling",
        f"common n        : {sub_report.attrs['common_n']} cells per level "
        f"(per-donor cap {sub_report.attrs['donor_cap']}) -> {run_ad.n_obs:,} cells",
        f"scored edges    : {len(res):,}  (CTCL)   |  {len(res_hc):,}  (HC negative control)",
        f"HC control      : {hc_note}",
        f"expr_prop={cfg.EXPR_PROP}  min_cells={cfg.MIN_CELLS}  seed={cfg.SEED}",
        "magnitude_rank / cellphone_pvals: deleted from the frame, absence asserted.",
        "full frame -> tables/ccc40_run.parquet (CSV holds the top 5,000 rows).",
    ],
    tables={"realised n per level": sub_report, "top 15 edges by specificity_rank": top},
)
```

```
common n = 2423 (n_raw 2846 on LC, per-donor cap 569) -> 41,191 cells over 17 levels
spec_ctcl: 60,404 edges scored at expr_prop=0.1 over 17 sender levels, 272 ordered level pairs
spec_hc: 6,466 edges scored at expr_prop=0.1 over 6 sender levels, 30 ordered level pairs
==============================================================================================================
SECTION 1 -- ONE RUN, PERMUTATION-FREE SPECIFICITY CONSENSUS
==============================================================================================================
methods         : NATMI + Connectome + log2FC, RRA aggregate, rank = specificity_rank
grid            : 17 levels, 272 ordered level pairs, both directions, ONE run
window          : CTCL (MF, SS, CTCL_other) applied BEFORE subsampling
common n        : 2423 cells per level (per-donor cap 569) -> 41,191 cells
scored edges    : 60,404  (CTCL)   |  6,466  (HC negative control)
HC control      : 6 levels, 2,304 cells, no CD4 level exists in HC skin
expr_prop=0.1  min_cells=25  seed=1337
magnitude_rank / cellphone_pvals: deleted from the frame, absence asserted.
full frame -> tables/ccc40_run.parquet (CSV holds the top 5,000 rows).

--- realised n per level ---
         level  n_window  n_after_donor_cap  n_final  n_donors_final  power_deficient
  Keratinocyte     83919              20529     2423              64            False
 CD4_malignant     76349              13999     2423              31            False
  CD4_reactive     71488              12337     2423              33            False
           CD8     51531              19718     2423              69            False
           cDC     29952              20285     2423              71            False
             B     19377               9686     2423              59            False
      DC_LAMP3     19061               8101     2423              65            False
   F_reticular     18522              11646     2423              65            False
F_inflammatory     14131               5135     2423              42            False
     Mac_FOLR2     13543              12207     2423              69            False
      Mac_infl     12720               6932     2423              61            False
       F_apCAF     10043               7831     2423              60            False
     Mono_moDC      7017               5564     2423              68            False
   F_papillary      7006               6673     2423              65            False
 F_mesenchymal      4933               4502     2423              56            False
Mac_SPP1_TREM2      3809               2831     2423              67            False
            LC      2846               2423     2423              69            False

--- top 15 edges by specificity_rank ---
        source         target ligand_complex receptor_complex  specificity_rank  expr_prod
           CD8   Keratinocyte           CCL5             SDC1         4.892e-10      2.173
      Mac_infl              B          CXCL8            CD79A         5.836e-10      6.238
             B Mac_SPP1_TREM2          FCER2      ITGAX_ITGB2         8.712e-10      1.249
       F_apCAF              B             C3             CD19         5.295e-09      2.415
 F_mesenchymal              B            FN1            CD79A         5.743e-09      5.136
      Mac_infl   Keratinocyte          CXCL8             SDC1         8.068e-09       1.83
      DC_LAMP3   CD4_reactive          CCL22             CCR4         1.445e-08      1.461
F_inflammatory       DC_LAMP3           CSF3           CSF2RA         1.715e-08      1.074
 CD4_malignant             LC           IL22          IL22RA2          2.07e-08      0.256
     Mac_FOLR2            CD8          CCL13            CXCR3          2.07e-08     0.9364
           CD8        F_apCAF           CCL5           ADRA2A         3.131e-08      1.381
F_inflammatory        F_apCAF            IL6             MUSK         3.202e-08     0.3562
      Ma
... [264 more chars truncated]
```

## §2 · Donor reproducibility — the primary evidence axis

Per edge: among donors where **both** levels clear `MIN_CELLS`, the fraction where **both** the
ligand complex clears `expr_prop` in the sender and the receptor complex clears it in the
receiver. A complex is only as good as its worst subunit. Reported with a Wilson 95 % CI and
the number of distinct studies that recovered it.

This is the replicate-aware statement the cell-unit permutation p never was. The cube is
regenerated for this roster — `ccc_skin_pseudobulk_full.parquet` is keyed on the v1 pooled
levels × sample and carries no detection counts. Built from the object already in memory; the
48 GB source atlas is not re-read. It is built on the **windowed, un-subsampled** object: the
subsample exists to equalise the ranking, not to throw away donor evidence.

```python  # cell 9
cube, meta = U.build_pseudobulk_cube(ctcl)
repro = U.donor_reproducibility(res, cube, meta)
repro.to_csv(U.tab("donor_reproducibility"), index=False)

REPRO_OK = repro[repro["repro_ok"]][U.KEY_COLS]
top_r = (repro.sort_values(["donor_frac", "n_studies_recovered"], ascending=False)
         .head(15)[U.KEY_COLS + ["n_donors_tested", "n_donors_recovered", "donor_frac",
                                 "donor_frac_lo", "donor_frac_hi", "n_studies_recovered"]])
U.printout(
    "SECTION 2 -- DONOR REPRODUCIBILITY (C5a, primary axis)",
    lines=[
        f"cube            : {cube.shape[0]:,} (level, donor, gene) rows over "
        f"{meta['level'].nunique()} levels and {meta['donor'].nunique()} donors",
        f"rule            : donor counts if BOTH levels have >= {cfg.MIN_CELLS} cells; edge is "
        f"recovered if every ligand subunit clears expr_prop={cfg.EXPR_PROP} in the sender and "
        "every receptor subunit in the receiver",
        f"edges tested    : {len(repro):,}",
        f"gate            : donor_frac >= {U.MIN_DONOR_FRAC}, >= {U.MIN_STUDIES} studies, "
        f">= {U.MIN_DONORS_REPRO} donors tested",
        f"edges passing   : {len(REPRO_OK):,} ({len(REPRO_OK) / max(len(repro), 1):.1%})",
        f"median donor_frac over tested edges: {repro['donor_frac'].median():.3f}",
    ],
    tables={"most reproducible edges": top_r},
)
```

```
cube (778562, 5) -> ccc_skin_pseudobulk_sub.parquet; meta (1144, 7) -> ccc_skin_pseudobulk_sub_meta.parquet
60,404 edges | 42,673 clear donor_frac>=0.5 in >=2 studies with >=5 donors tested
==============================================================================================================
SECTION 2 -- DONOR REPRODUCIBILITY (C5a, primary axis)
==============================================================================================================
cube            : 778,562 (level, donor, gene) rows over 18 levels and 73 donors
rule            : donor counts if BOTH levels have >= 25 cells; edge is recovered if every ligand subunit clears expr_prop=0.1 in the sender and every receptor subunit in the receiver
edges tested    : 60,404
gate            : donor_frac >= 0.5, >= 2 studies, >= 5 donors tested
edges passing   : 42,673 (70.6%)
median donor_frac over tested edges: 0.667

--- most reproducible edges ---
   source    target ligand_complex receptor_complex  n_donors_tested  n_donors_recovered  donor_frac  donor_frac_lo  donor_frac_hi  n_studies_recovered
Mono_moDC       CD8            VIM             CD44               43                  43           1          0.918              1                    7
      CD8 Mono_moDC            VIM             CD44               43                  43           1          0.918              1                    7
      cDC Mac_FOLR2            VIM             CD44               55                  55           1         0.9347              1                    7
Mac_FOLR2       cDC            VIM             CD44               55                  55           1         0.9347              1                    7
      CD8 Mono_moDC            MIF        CD44_CD74               43                  43           1          0.918              1                    7
Mono_moDC       CD8            MIF        CD44_CD74               43                  43           1          0.918              1                    7
Mono_moDC       CD8            PKM             CD44               43                  43           1          0.918              1                    7
      cDC Mac_FOLR2          TIMP1             CD63               55                  55           1         0.9347              1                    7
      CD8 Mono_moDC            PKM             CD44               43                  43           1          0.918              1                    7
      cDC Mac_FOLR2            PKM             CD44               55                  55           1         0.9347              1                    7
      CD8 Mono_moDC            MIF       CD74_CXCR4               43                  43           1          0.918              1                    7
Mac_FOLR2       cDC           CD14            ITGB1               55                  55           1         0.9347              1                    7
      CD8 Mono_moDC         LGALS1            PTPRC               43                  43           1          0.918              1                    7
Mono_moDC       CD8          TIMP1             CD63               43                  43           1          0.918              1                    7
Mono_moDC       CD8         S100A8             CD69               43                  43           1          0.918              1                    7
```

## §3 · Malignant vs reactive CD4, paired within donor

Replaces `rank_delta` and `malignant_only` outright. `rank_delta` subtracted RRA scores from
two runs on different residual pools and different donor sets; `malignant_only` promoted a
coverage failure — the pair simply not scored in the reactive run — to evidence.

Here: donors carrying ≥25 of **both** CD4 levels, pseudobulk, design `~ donor + celltype`, so
donor, study, chemistry and stage all cancel. This is the one contrast in this atlas free of
study confounding (`ccc_data.CONTRASTS["malignant_vs_reactive"]`). BH-FDR across the grid.

Fitted with **PyDESeq2** — there is no R and no rpy2 in this environment, so the spec's
edgeR/limma-voom is unavailable. Size factors come from the 1,832 resource genes only;
median-of-ratios is robust to that, but it is a real limitation and is logged as one.

Per edge the malignant-side gene is tested: the ligand where malignant CD4 sends, the receptor
where it receives, worst subunit for a complex. `contrast_ok` = FDR < 0.05 **and** up in
malignant.

```python  # cell 11
de_genes, contrast = U.paired_contrast(cube, meta, res)
de_genes.to_csv(U.tab("paired_contrast_genes"), index=False)
contrast.to_csv(U.tab("paired_contrast_edges"), index=False)

sig = de_genes[(de_genes["fdr"] < U.FDR_ALPHA)]
top_g = de_genes.sort_values("fdr").head(15)[["gene", "lfc", "fdr"]]
U.printout(
    "SECTION 3 -- MALIGNANT vs REACTIVE CD4, PAIRED WITHIN DONOR (C5b)",
    lines=[
        f"engine          : PyDESeq2 (no R/rpy2 in this env; substituted for edgeR/limma-voom)",
        f"design          : ~ donor + celltype   contrast {cfg.CD4_MALIGNANT} vs {cfg.CD4_REACTIVE}",
        f"donors          : {contrast.attrs['n_donors']} with >= {cfg.MIN_CELLS} cells of BOTH "
        "levels (config records n_units=26; the measured value is what is used)",
        f"genes tested    : {len(de_genes):,} of {len(cube['gene'].unique()):,} in the cube",
        f"genes at FDR<{U.FDR_ALPHA}: {len(sig):,} ({int((sig['lfc'] > 0).sum()):,} up in malignant)",
        f"edges with the malignant side up at FDR<{U.FDR_ALPHA}: "
        f"{int(contrast['contrast_ok'].sum()):,} / {len(contrast):,}",
        "caveat: size factors are estimated on the 1,832 resource genes, not the full "
        "transcriptome; median-of-ratios is robust to this but it is not the same fit.",
    ],
    tables={"top genes by FDR": top_g},
)
```

```
paired contrast: 29 donors, 58 pseudo-samples, 1303 genes (of 1755 in the object)
871/60,404 edges: malignant side up at FDR<0.05
==============================================================================================================
SECTION 3 -- MALIGNANT vs REACTIVE CD4, PAIRED WITHIN DONOR (C5b)
==============================================================================================================
engine          : PyDESeq2 (no R/rpy2 in this env; substituted for edgeR/limma-voom)
design          : ~ donor + celltype   contrast CD4_malignant vs CD4_reactive
donors          : 29 with >= 25 cells of BOTH levels (config records n_units=26; the measured value is what is used)
genes tested    : 1,303 of 1,820 in the cube
genes at FDR<0.05: 303 (107 up in malignant)
edges with the malignant side up at FDR<0.05: 871 / 60,404
caveat: size factors are estimated on the 1,832 resource genes, not the full transcriptome; median-of-ratios is robust to this but it is not the same fit.

--- top genes by FDR ---
    gene    lfc       fdr
    CD8A -5.982 7.116e-28
     CD7 -5.074 1.904e-27
  PTGER2  -4.93 9.456e-24
    PCNA  1.161 1.966e-19
LGALS3BP -3.548 6.719e-15
   PTHLH  2.871  2.45e-13
   ITGA6 -2.293 1.091e-12
    CCR6 -3.359 7.631e-12
   ITGA4 -2.882 9.759e-12
   KLRB1 -2.941 2.103e-11
  CXCL13  2.993 1.137e-10
  FLT3LG  -1.21  1.38e-10
    CCR5 -2.773 3.909e-10
   PTGDR -3.481 3.909e-10
   GPR25 -4.724 5.364e-10
```

## §4 · Where each named edge lands — power-equalised within donor

`magnitude_rank` and `expr_prop` both depend on detection rate, which depends on cells per level
and on sequencing depth. Ranking levels against each other with no power equalisation measured
n as much as biology, and the old `gap_rank_ratio < 3` / `spread_out` verdict had no null at
all. Both are deleted.

Replacement: **within each donor**, downsample every candidate level that clears `MIN_CELLS` to
the smallest such level *in that donor*, run the consensus once over the candidate grid, and
rank the candidate levels against each other. The comparison is then always inside one run and
at equal n. Across donors: **Skillings–Mack** (Friedman when no level is missing) plus a
**donor jackknife** — the percentage of leave-one-donor-out runs in which the winner still wins.

A level that is not claimable in a donor is missing, not zero — which is exactly what
Skillings–Mack is for.

`pDC` **is** a candidate here even though it is out of the ranked run: this test equalises power
within donor by construction, so the reason pDC was dropped from §1 does not apply.

```python  # cell 13
land_ranks, land = U.landing_test(ctcl, resource)
land_ranks.to_csv(U.tab("landing_ranks"), index=False)
land.to_csv(U.tab("landing_summary"), index=False)

win = land[land["is_winner"]] if len(land) else land
lines = [
    f"tests           : {len(cfg.RESOLUTION_TESTS)} named edges "
    f"({', '.join(cfg.RESOLUTION_TESTS)})",
    f"design          : within donor, every candidate level downsampled to the smallest "
    f"claimable n in that donor, one consensus run per donor",
    f"donors scored   : {land_ranks['donor'].nunique() if len(land_ranks) else 0}",
    "gap_rank_ratio and the uniform/concentrated verdict: DELETED (no null).",
]
for t in cfg.RESOLUTION_TESTS:
    w = win[win["test"] == t]
    if not len(w):
        lines.append(f"  {t:<12} not scored in any donor -- coverage, not biology")
        continue
    w = w.iloc[0]
    exp = cfg.RESOLUTION_TESTS[t]["expected"]
    lines.append(
        f"  {t:<12} winner {w['level']:<16} median rank {w['median_within_donor_rank']:.1f} "
        f"over {int(w['n_donors'])} donors | jackknife {100 * w['jackknife_win_pct']:.0f}% | "
        f"Skillings-Mack p={w['skillings_mack_p']:.3g} | expected {exp} -> "
        f"{'AS EXPECTED' if w['level'] in exp else 'NOT as expected'}")
U.printout("SECTION 4 -- EDGE LANDING, POWER-EQUALISED (C5c)", lines=lines,
           tables={"per (test, level)": land})
```

```
donor B5__PT09: 12 candidates at n=31
  donor B5__PT23: 9 candidates at n=26
  donor D1__P171: 7 candidates at n=30
  donor D1__P204: 7 candidates at n=32
  donor D1__P311: 10 candidates at n=25
  donor D1__P312: 9 candidates at n=29
  donor D1__P76: 7 candidates at n=76
  donor D1__PGS: 8 candidates at n=29
  donor D3__P105: 7 candidates at n=31
  donor D3__P192: 10 candidates at n=37
  donor D3__P196: 9 candidates at n=42
  donor D5__MFIVB: 7 candidates at n=28
  donor Li2024_atlas__CTCL1: 11 candidates at n=43
  donor Li2024_atlas__CTCL2: 5 candidates at n=86
  donor Li2024_atlas__CTCL3: 10 candidates at n=32
  donor Li2024_atlas__CTCL4: 9 candidates at n=28
  donor Li2024_atlas__CTCL5: 12 candidates at n=33
  donor Li2024_atlas__CTCL6: 12 candidates at n=38
  donor Li2024_atlas__CTCL7: 11 candidates at n=39
  donor Li2024_atlas__CTCL8: 12 candidates at n=69
  donor Li2024_atlas__PT11: 9 candidates at n=30
  donor Li2024_atlas__PT35: 4 candidates at n=75
  donor Li2024_atlas__PT50: 3 candidates at n=96
  donor Li2024_atlas__PT56: 8 candidates at n=25
==============================================================================================================
SECTION 4 -- EDGE LANDING, POWER-EQUALISED (C5c)
==============================================================================================================
tests           : 5 named edges (ccr4_axis, csf1_axis, cxcl12_axis, mhcii_axis, tgfb_axis)
design          : within donor, every candidate level downsampled to the smallest claimable n in that donor, one consensus run per donor
donors scored   : 24
gap_rank_ratio and the uniform/concentrated verdict: DELETED (no null).
  ccr4_axis    winner DC_LAMP3         median rank 1.0 over 14 donors | jackknife 100% | Skillings-Mack p=0.000449 | expected ['cDC', 'DC_LAMP3'] -> AS EXPECTED
  csf1_axis    not scored in any donor -- coverage, not biology
  cxcl12_axis  winner F_reticular      median rank 1.0 over 21 donors | jackknife 100% | Skillings-Mack p=2.41e-08 | expected ['F_inflammatory'] -> NOT as expected
  mhcii_axis   winner cDC              median rank 1.0 over 10 donors | jackknife 50% | Skillings-Mack p=0.00113 | expected ['F_apCAF'] -> NOT as expected
  tgfb_axis    winner F_papillary      median rank 1.0 over 9 donors | jackknife 94% | Skillings-Mack p=0.188 | expected ['F_mesenchymal'] -> NOT as expected

--- per (test, level) ---
       test          level  median_within_donor_rank  n_donors  is_winner  expected  jackknife_win_pct  skillings_mack_stat  skillings_mack_df  skillings_mack_p
  ccr4_axis       DC_LAMP3                         1        14       True      True                  1                26.28                  7         0.0004491
  ccr4_axis Mac_SPP1_TREM2                         2         6      False     False                NaN                26.28                  7         0.0004491
  ccr4_axis      Mac_FOLR2                       2.5         8      False     False                NaN                26.28                  7         0.0004491
  ccr4_axis            cDC                         3        12      False      True                NaN                26.28                  7         0.0004491
  ccr4_axis             LC                         4         7      False     False                NaN                26.28                  7         0.0004491
  ccr4_axis       Mac_infl                         4         8      False     False                NaN                26.28                  7         0.0004491
  ccr4_axis      Mono_moDC                         4         5      False     False                NaN                26.28                  7         0.0004491
  ccr4_axis            pDC                         4         4      False     False                NaN                26.28                  7         0.0004491
cxcl12_axis    F_reticular                         1        21       True     False                  1                41.23                  4         2.407e-08
... [2576 more chars truncated]
```

## §5 · Controls

- **positive / negative controls** — rank position on `specificity_rank`; where a control is
  missing, `grid_coverage` says whether it was a coverage failure or a genuine absence.
- **`SPEC_MUST_HAVES`** — the three edges the replication spec names, all on the **untouched**
  `B` level, so they are a regression test against v1 rather than a new result.
- **within-donor label shuffle** — holds each donor's composition fixed, so what dissolves is
  cell-type specificity rather than donor identity. The per-seed top-20 overlap is *commentary*:
  it is a max over `N_SHUFFLES = 3` seeds and moves with seed noise. The **gate** is per edge —
  any edge that reaches a shuffled top-20 is recoverable with cell-type identity destroyed, so it
  is abundance-driven and section 7 drops it (gate 5). Same move as the ambient arm below:
  a control that names the offending edges beats a control that returns a pass/fail count.
- **ambient regression** — replaces the `LR_CAVEATS` blacklist, which was dropping exactly the
  IL4/IL13 pairs the forced panel exists to report. Per edge, the donor-level score is regressed
  on that donor's malignant fraction and log median library size; an edge explained by the
  malignant fraction is flagged, not deleted.
- **`evidence == 'both'` arm** — `CD4_malignant` restricted to cells that also carry a CNV call,
  a nested subset under the ALICE primary. The common-n design makes this power-matched by
  construction: both arms run at the same cells per level, so a lost edge is a lost edge and not
  lost power.
- **doublet arm — not computable on this atlas.** `doublet_score` is null for 100 % of li2024,
  which is 56 % of skin cells, and where it exists it maxes at 0.249. A filter would remove
  ~0.5 % of the cells that have a score and would silently be a li2024-vs-rest contrast.

```python  # cell 15
  import importlib; importlib.reload(U)
```

```
<module 'ccc_utils' from '/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/ccc_utils.py'>
```

```python  # cell 16
ctrl = U.control_report(res, resource=resource)
missing = ctrl[~ctrl["found"]]
why = U.grid_coverage(run_ad, resource, missing[U.KEY_COLS]) if len(missing) else pd.DataFrame()

must = pd.DataFrame(U.resolve_controls(cfg.SPEC_MUST_HAVES, resource), columns=U.KEY_COLS)
must = must.merge(res[U.KEY_COLS + [U.RANK_COL, "specificity_pct"]], on=U.KEY_COLS, how="left")
must["present"] = must[U.RANK_COL].notna()

shuf, shuf_edges = U.shuffle_control(run_ad, resource, pairs, reference=res)
shuf_edges.to_csv(U.tab("shuffle_recurrent_edges"), index=False)

amb = U.ambient_regression(res, cube, meta, edges=REPRO_OK)
amb.to_csv(U.tab("ambient_regression"), index=False)

ev_keep = ((ctcl.obs[cfg.GROUPBY].astype(str) != cfg.CD4_MALIGNANT)
           | (ctcl.obs[cfg.EVIDENCE_SRC].astype(str) == "both"))
both_ad, both_report = U.subsample_common_n(ctcl[ev_keep.values].copy(), MAIN_ROSTER, verbose=False)
res_both = U.run_spec_consensus(both_ad, resource, pairs, key_added="spec_both", verbose=False)
ov_both = len(set(map(tuple, U.top_edges(res, cfg.TOP_N)[U.KEY_COLS].values))
              & set(map(tuple, U.top_edges(res_both, cfg.TOP_N)[U.KEY_COLS].values)))

ctrl.to_csv(U.tab("controls"), index=False)
is_pos = ctrl["kind"] == "positive"
n_pos, n_pos_found = int(is_pos.sum()), int(ctrl.loc[is_pos, "found"].sum())
n_neg, n_neg_found = int((~is_pos).sum()), int(ctrl.loc[~is_pos, "found"].sum())
n_rec = int(shuf_edges["shuffle_recurrent"].sum()) if len(shuf_edges) else 0
n_rec_real = int((shuf_edges["shuffle_recurrent"] & shuf_edges["in_real_top_n"]).sum()) if len(shuf_edges) else 0
U.printout(
    "SECTION 5 -- CONTROLS",
    lines=[
        f"positives found : {n_pos_found}/{n_pos}",
        f"negatives found : {n_neg_found}/{n_neg}  "
        "(lineage-impossible; want 0 or bottom decile)",
        f"SPEC_MUST_HAVES : {int(must['present'].sum())}/{len(must)} present on the untouched "
        "B level (regression test against v1)",
        f"within-donor shuffle: overlap with the real top-{cfg.TOP_N} = "
        f"{shuf['overlap_with_real'].tolist()} across {cfg.N_SHUFFLES} seeds -- COMMENTARY, "
        "not a pass/fail",
        f"shuffle-recurrent edges: {n_rec} distinct edges reach a shuffled top-{cfg.TOP_N} "
        f"({n_rec_real} of them also in the real top-{cfg.TOP_N}); all {n_rec} are gated out "
        "in section 7 (gate 5)",
        f"ambient         : {int(amb['ambient_flag'].sum()):,}/{len(amb):,} reproducible edges "
        f"explained by donor malignant fraction (FDR<{U.FDR_ALPHA}, positive slope)",
        f"evidence=='both': {both_ad.n_obs:,} cells at the same common n, top-{cfg.TOP_N} "
        f"overlap with the primary = {ov_both}/{cfg.TOP_N} (power-matched by construction)",
        "LR_CAVEATS      : NOT a gate here. Printed as commentary in section 6 only.",
        "doublet arm     : NOT COMPUTABLE. doublet_score is null for 100% of li2024 (56% of "
        "skin cells) and maxes at 0.249 elsewhere; a filter would be a study contrast.",
    ],
    tables={"controls": ctrl, "why a missing control was not scored": why,
            "spec must-haves": must,
            "shuffle-recurrent edges (gated out in section 7)": shuf_edges,
            "top ambient-flagged edges": amb.sort_values("fdr_malig_frac").head(10)},
)
```

```
0/42,673 edges explained by donor malignant fraction (FDR<0.05, positive slope)
==============================================================================================================
SECTION 5 -- CONTROLS
==============================================================================================================
positives found : 10/12
negatives found : 0/3  (lineage-impossible; want 0 or bottom decile)
SPEC_MUST_HAVES : 2/3 present on the untouched B level (regression test against v1)
within-donor shuffle: overlap with the real top-20 = [2, 4, 2] across 3 seeds -- COMMENTARY, not a pass/fail
shuffle-recurrent edges: 26 distinct edges reach a shuffled top-20 (4 of them also in the real top-20); all 26 are gated out in section 7 (gate 5)
ambient         : 0/42,673 reproducible edges explained by donor malignant fraction (FDR<0.05, positive slope)
evidence=='both': 41,191 cells at the same common n, top-20 overlap with the primary = 19/20 (power-matched by construction)
LR_CAVEATS      : NOT a gate here. Printed as commentary in section 6 only.
doublet arm     : NOT COMPUTABLE. doublet_score is null for 100% of li2024 (56% of skin cells) and maxes at 0.249 elsewhere; a filter would be a study contrast.

--- controls ---
    kind         source         target ligand_complex    receptor_complex  found  rank_position  specificity_rank  pct_of_scored in_top_n
positive  CD4_malignant              B         CXCL13               CXCR5   True             35         3.085e-07      0.0005794    False
positive  CD4_malignant              B         CD40LG                CD40   True           4555          0.008062        0.07541    False
positive              B  CD4_malignant           CD86                CD28  False            NaN               NaN            NaN      NaN
positive              B  CD4_malignant            TNF            TNFRSF1B   True      2.883e+04            0.5174         0.4772    False
positive            cDC  CD4_malignant           CD86                CD28   True           9559           0.03998         0.1583    False
positive            cDC  CD4_malignant          CCL17                CCR4   True      1.175e+04           0.06323         0.1945    False
positive       DC_LAMP3  CD4_malignant          CCL22                CCR4   True             40         3.566e-07      0.0006622    False
positive      Mac_FOLR2  CD4_malignant           CD58                 CD2   True      2.385e+04            0.3246         0.3949    False
positive F_inflammatory  CD4_malignant         CXCL12               CXCR4   True       2.02e+04            0.2208         0.3344    False
positive  CD4_malignant Mac_SPP1_TREM2      CSF1_IL34               CSF1R  False            NaN               NaN            NaN      NaN
positive  CD4_malignant  F_mesenchymal          TGFB1       TGFBR1_TGFBR2   True      2.114e+04            0.2439           0.35    False
positive  CD4_malignant  F_mesenchymal          TGFB1 ACVR1_TGFBR1_TGFBR2   True      1.848e+04            0.1796          0.306    False
negative  CD4_malignant    F_reticular          KITLG                 KIT  False            NaN               NaN            NaN      NaN
negative  CD4_malignant   Keratinocyte         COL1A1         ITGA2_ITGB1  False            NaN               NaN            NaN      NaN
negative  CD4_malignant   Keratinocyte           KRT1                   *  False            NaN               NaN            NaN      NaN

--- why a missing control was not scored ---
       source         target ligand_complex receptor_complex  n_cells_source  n_cells_target  ligand_prop_source  receptor_prop_target                 reason
            B  CD4_malignant           CD86             CD28            2423            2423             0.08337                0.2542 ligand_below_expr_prop
CD4_malignant Mac_SPP1_TREM2      CSF1_IL34            CSF1R            2423            2423           0.0008254                0.6859 ligand_below_expr_prop
CD4_malignant    F_retic
... [5448 more chars truncated]
```

## §6 · Forced curated panel

The 49 curated pairs are reported **whether or not they cleared `expr_prop`** — reporting the
negatives is the point. Three distinct reasons a curated pair can be missing, kept apart because
they look identical in a result table:

1. **absent from the resource** — can never be scored however well expressed;
2. **genes absent from the data**;
3. **below `expr_prop` / `min_cells`** in this window — a coverage limit, not a biological
   negative.

`LR_CAVEATS` is printed here as commentary. It gates nothing: as a blacklist it was deleting the
IL4/IL13 biology that this panel exists to report.

```python  # cell 18
panel = U.resolve_panel(resource=resource)
pcov = U.panel_coverage(resource, var_names=run_ad.var_names)
fpe = U.forced_panel_expression(run_ad, resource=resource)

panel_edges = (res.merge(panel[["group", "ligand_complex", "receptor_complex"]].dropna(),
                         on=["ligand_complex", "receptor_complex"], how="inner"))
panel_edges.to_csv(U.tab("panel_scored"), index=False)
pcov.to_csv(U.tab("panel_coverage"), index=False)
fpe.to_csv(U.tab("panel_detection"), index=False)

focal_panel = panel_edges[(panel_edges.source == cfg.CD4_MALIGNANT)
                          | (panel_edges.target == cfg.CD4_MALIGNANT)]
not_scored = pcov[pcov["in_resource"] & pcov["in_data"]].merge(
    panel_edges[["ligand_complex", "receptor_complex"]].drop_duplicates(),
    on=["ligand_complex", "receptor_complex"], how="left", indicator=True)
not_scored = not_scored[not_scored["_merge"] == "left_only"].drop_duplicates(
    ["group", "ligand_in", "receptor_in"])

print("LR_CAVEATS (commentary only -- gates nothing):")
for gene, note in cfg.LR_CAVEATS.items():
    print(f"  {gene}: {note}")

U.printout(
    "SECTION 6 -- FORCED CURATED PANEL",
    lines=[
        f"curated pairs   : {len(pcov)} rows, {int(pcov['in_resource'].sum())} resolve to a "
        f"resource interaction, {int(pcov['in_data'].sum())} have every gene in the data",
        f"absent from the resource (never scorable): "
        f"{int((~pcov['in_resource']).sum())}",
        f"genes absent from the data: "
        f"{int((pcov['in_resource'] & ~pcov['in_data']).sum())}",
        f"in resource + in data but BELOW expr_prop/min_cells here: {len(not_scored)}",
        f"panel edges scored in this run: {len(panel_edges):,} "
        f"({len(focal_panel):,} touching {cfg.CD4_MALIGNANT})",
        "detection proportions per (level, gene) -> tables/ccc40_panel_detection.csv, so a "
        "negative reads as 'detected in 6% of the sender' rather than as absence.",
    ],
    tables={"panel edges on the malignant axis (top 15)":
                focal_panel.nsmallest(15, U.RANK_COL)[
                    ["group"] + U.KEY_COLS + [U.RANK_COL]],
            "curated pairs not scored here":
                not_scored[["group", "ligand_in", "receptor_in"]].head(20)},
)
```

```
LR_CAVEATS (commentary only -- gates nothing):
  MIF: MIF is ubiquitous and very highly expressed; a top-ranked MIF->CD74 or MIF->CXCR4 edge is a null-model failure until shown otherwise. Treat it as the positive control for the FAILURE mode, not as biology.
  B2M: Housekeeping-level expression; any B2M edge is an artifact of the null.
  HLA-DRA: MHC-II 'interactions' in the consensus resource largely track myeloid identity rather than a signalling event.
  CD74: See MIF and HLA-DRA -- CD74 is the receiver side of both failure modes.
  IL13: Th2 cytokine mRNA is poorly captured by 3' 10x, so a negative IL13->IL4R edge is uninformative. That is exactly why it is in the forced panel.
  IL4: See IL13.
  TNFSF8: CD30L->CD30 marks large-cell transformation; expect it only in advanced or LCT donors, and read its absence as coverage, not biology.
  PVR: The consensus resource stores the nectin axis with PVR on the RECEPTOR side (TIGIT -> PVR, CD96 -> PVR), which is backwards biologically: TIGIT and CD96 are the receptors. ccc_helpers.resolve_pairs flips the direction to match the resource, so read these edges as 'the PVR-expressing cell is liana's target' and interpret the biology in the opposite direction.
  TIGIT: See PVR -- direction is stored inverted in the resource.
==============================================================================================================
SECTION 6 -- FORCED CURATED PANEL
==============================================================================================================
curated pairs   : 58 rows, 58 resolve to a resource interaction, 58 have every gene in the data
absent from the resource (never scorable): 0
genes absent from the data: 0
in resource + in data but BELOW expr_prop/min_cells here: 14
panel edges scored in this run: 1,502 (324 touching CD4_malignant)
detection proportions per (level, gene) -> tables/ccc40_panel_detection.csv, so a negative reads as 'detected in 6% of the sender' rather than as absence.

--- panel edges on the malignant axis (top 15) ---
                  group        source         target ligand_complex receptor_complex  specificity_rank
        b_cell_tls_axis CD4_malignant              B         CXCL13            CXCR5         3.085e-07
  chemokine_recruitment      DC_LAMP3  CD4_malignant          CCL22             CCR4         3.566e-07
  chemokine_recruitment      DC_LAMP3  CD4_malignant          CCL17             CCR4         6.477e-05
costim_tme_to_malignant CD4_malignant            CD8           CD70             CD27         0.0001473
   malignant_to_myeloid CD4_malignant Mac_SPP1_TREM2           CSF2    CSF2RA_CSF2RB         0.0004972
  chemokine_recruitment       F_apCAF  CD4_malignant          CXCL9            CXCR3         0.0005364
   checkpoint_restraint      DC_LAMP3  CD4_malignant           CD80            CTLA4         0.0005702
   checkpoint_restraint      DC_LAMP3  CD4_malignant           CD86            CTLA4         0.0005957
costim_tme_to_malignant      DC_LAMP3  CD4_malignant           CD80             CD28         0.0006514
        b_cell_tls_axis CD4_malignant       DC_LAMP3           CD28             CD80         0.0006514
costim_tme_to_malignant      DC_LAMP3  CD4_malignant           CD86             CD28         0.0006848
        b_cell_tls_axis CD4_malignant       DC_LAMP3           CD28             CD86         0.0006848
  chemokine_recruitment       F_apCAF  CD4_malignant          CCL19             CCR7         0.0007551
costim_tme_to_malignant CD4_malignant       DC_LAMP3           CD96              PVR          0.001249
        b_cell_tls_axis CD4_malignant       DC_LAMP3         CD40LG             CD40          0.001568

--- curated pairs not scored here ---
                  group ligand_in    receptor_in
costim_tme_to_malignant    ICOSLG           ICOS
costim_tme_to_malignant    TNFSF4        TNFRSF4
costim_tme_to_malignant   TNFSF18       TNFRSF18
costim_tme_to_malignant    TNFSF8        TNFRSF8
   checkpoint_restra
... [469 more chars truncated]
```

## §7 · The reportable set, and the figures

Six gates, ANDed. Every one of them is a donor-level or coverage statement; none is a
cell-unit p-value.

| gate | rule |
|---|---|
| 1 | both levels ∈ `CLAIMABLE` (≥5 donors at ≥25 cells, CTCL window) |
| 2 | `donor_frac` ≥ 0.5 in ≥2 studies (§2) |
| 3 | paired-contrast FDR < 0.05, malignant side up (§3) |
| 4 | not ambient-flagged (§5) |
| 5 | not recoverable under the within-donor label shuffle (§5) |
| 6 | `specificity_rank` in the top decile of the run |

Gate 5 is what the shuffle arm buys: an edge that reaches the shuffled top-20 is one abundance
alone can produce, so it is dropped by name rather than being counted against a threshold.

Then four figures, and that is the whole set: no network, chord or circos plot, no UMAP, no
violin or ridge plot, no per-section exploratory panel.

```python  # cell 20
reportable, drops = U.reportable_set(res, repro, contrast, amb, CLAIMABLE,
                                    shuffle_recurrent=shuf_edges)
reportable.to_csv(U.tab("reportable_set"), index=False)
drops.to_csv(U.tab("reportable_gates"), index=False)

neg = U.resolve_controls(cfg.NEGATIVE_CONTROLS, resource)
in_rep = set(map(tuple, reportable[U.KEY_COLS].values))
leaked = [n for n in neg if tuple(n) in in_rep]
assert not leaked, f"lineage-impossible negative controls in the reportable set: {leaked}"

rec = shuf_edges[shuf_edges["shuffle_recurrent"]] if len(shuf_edges) else shuf_edges
rec_leaked = [k for k in map(tuple, rec[U.KEY_COLS].values)] if len(rec) else []
rec_leaked = [k for k in rec_leaked if k in in_rep]
assert not rec_leaked, f"shuffle-recurrent edges survived gate 5: {rec_leaked}"

U.printout(
    "SECTION 7 -- REPORTABLE SET",
    lines=[
        f"scored edges in : {len(res):,}",
        f"reportable out  : {len(reportable):,}",
        f"partners represented: {sorted(set(reportable['source']) | set(reportable['target']))}",
        f"negative controls leaked into the set: {len(leaked)} (asserted 0)",
        f"shuffle-recurrent edges leaked into the set: {len(rec_leaked)} (asserted 0); "
        f"{int(drops.loc[drops['gate'] == '5_not_shuffle_recurrent', 'n_dropped'].iloc[0])} "
        "dropped at gate 5",
        f"within-donor shuffle overlap (commentary): {shuf['overlap_with_real'].tolist()}",
    ],
    tables={"per-gate drop counts (applied in order)": drops,
            "reportable set (top 25)": reportable.head(25)[
                U.KEY_COLS + [U.RANK_COL, "donor_frac", "n_studies_recovered",
                              "contrast_lfc", "contrast_fdr"]]},
)
```

```
reportable set: 22 of 60,404 scored edges (top-decile cut specificity_rank <= 0.0147)
==============================================================================================================
SECTION 7 -- REPORTABLE SET
==============================================================================================================
scored edges in : 60,404
reportable out  : 22
partners represented: ['B', 'CD4_malignant', 'CD4_reactive', 'CD8', 'F_apCAF', 'F_inflammatory', 'F_reticular', 'Keratinocyte', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_moDC']
negative controls leaked into the set: 0 (asserted 0)
shuffle-recurrent edges leaked into the set: 0 (asserted 0); 0 dropped at gate 5
within-donor shuffle overlap (commentary): [2, 4, 2]

--- per-gate drop counts (applied in order) ---
                   gate  n_in  n_out  n_dropped  n_failing_alone
     1_levels_claimable 60404  60404          0                0
   2_donor_reproducible 60404  42673      17731            17731
      3_paired_contrast 42673    626      42047            59533
          4_not_ambient   626    626          0                0
5_not_shuffle_recurrent   626    626          0               25
           6_top_decile   626     22        604            54363

--- reportable set (top 25) ---
        source         target ligand_complex receptor_complex  specificity_rank  donor_frac  n_studies_recovered  contrast_lfc  contrast_fdr
 CD4_malignant            CD8           CD70             CD27         0.0001473      0.8519                    6         1.198     6.565e-05
 CD4_malignant              B           CD70        TNFRSF13B         0.0005791        0.65                    3         1.198     6.565e-05
 CD4_malignant      Mac_FOLR2            SPN          SIGLEC1         0.0006214        0.95                    5        0.5567         5e-06
     Mac_FOLR2  CD4_malignant           C1QB            C1QBP           0.00121           1                    5        0.4471       0.02128
 CD4_malignant   CD4_reactive           CD70             CD27          0.001995      0.8621                    6         1.198     6.565e-05
 CD4_malignant F_inflammatory        TNFSF14         TNFRSF6B          0.002061      0.7778                    2        0.9809     2.799e-06
 CD4_malignant   Keratinocyte         S100A4            ERBB3          0.002342           1                    6        0.4599       0.03163
F_inflammatory  CD4_malignant          ICAM1            IL2RG          0.003061           1                    2        0.2829      0.004094
 CD4_malignant        F_apCAF        TNFSF14         TNFRSF6B          0.003239      0.6842                    4        0.9809     2.799e-06
 CD4_malignant Mac_SPP1_TREM2            SPN          SIGLEC1          0.003637       0.625                    4        0.5567         5e-06
 CD4_malignant   Keratinocyte          HMGB1             SDC1          0.003963      0.5909                    5        0.5093       0.03946
 CD4_malignant   Keratinocyte          PTHLH            ADRB2          0.004768         0.5                    5         2.871      2.45e-13
 CD4_malignant F_inflammatory            MIF            ACKR3          0.007282      0.7778                    2        0.6899      1.08e-09
 CD4_malignant   Keratinocyte         S100A4            ERBB2          0.009525      0.7273                    5        0.4599       0.03163
 CD4_malignant       Mac_infl           CD99            PILRA           0.01025           1                    5        0.4355      0.002484
 CD4_malignant F_inflammatory          GNAI2            EDNRB           0.01062           1                    2        0.4335     0.0006806
     Mono_moDC  CD4_malignant          HBEGF             CD82           0.01068        0.75                    6         0.572      0.001353
 CD4_malignant    F_reticular            MIF            ACKR3           0.01093      0.7619                    5        0.6899      1.08e-09
 CD4_malignant Mac_SPP1_TREM2 
... [533 more chars truncated]
```

```python  # cell 21
f1 = U.plot_dotplot(res, repro, focal=cfg.CD4_MALIGNANT, partners=MAIN_ROSTER,
                    out=U.fig_path("dotplot"))
f2 = U.plot_heatmap(res, contrast, focal=cfg.CD4_MALIGNANT, partners=MAIN_ROSTER,
                    out=U.fig_path("heatmap"))
f3 = U.plot_landing_heatmap(land, out=U.fig_path("landing"))
f4 = U.plot_panel_dotplot(fpe, panel, levels=MAIN_ROSTER, out=U.fig_path("panel"))

U.printout(
    "SECTION 7 -- FIGURES",
    lines=[
        f"1 {U.fig_path('dotplot').name} : x = partner, y = interaction, colour = "
        "-log10 specificity_rank, size = fraction of donors recovering the edge; one panel "
        "per direction, top 20.",
        f"2 {U.fig_path('heatmap').name} : interactions x partner, shared vmin/vmax, grey = "
        f"structurally absent, * = paired-contrast FDR < {U.FDR_ALPHA}, top 15.",
        f"3 {U.fig_path('landing').name} : RESOLUTION_TEST x level, median within-donor rank, "
        "winner annotated with the donor-jackknife win %.",
        f"4 {U.fig_path('panel').name}   : forced curated panel, size = DETECTION proportion.",
        "no network / chord / circos, no UMAP, no violin or ridge plot.",
    ],
)
```

```
==============================================================================================================
SECTION 7 -- FIGURES
==============================================================================================================
1 ccc40_dotplot.svg : x = partner, y = interaction, colour = -log10 specificity_rank, size = fraction of donors recovering the edge; one panel per direction, top 20.
2 ccc40_heatmap.svg : interactions x partner, shared vmin/vmax, grey = structurally absent, * = paired-contrast FDR < 0.05, top 15.
3 ccc40_landing.svg : RESOLUTION_TEST x level, median within-donor rank, winner annotated with the donor-jackknife win %.
4 ccc40_panel.svg   : forced curated panel, size = DETECTION proportion.
no network / chord / circos, no UMAP, no violin or ridge plot.

<Figure size 1471.25x1707.2 with 3 Axes>
<Figure size 1529x1342 with 3 Axes>
<Figure size 1622.5x704 with 2 Axes>
<Figure size 1457.5x3586 with 2 Axes>
```

### Figure 1b · the same dots, grouped into biological programs

Identical encoding to figure 1 — x = partner, colour = `-log10 specificity_rank`, size = donor
fraction — but the rows are the curated edges of `U.CCC_PROGRAMS`, blocked by program and
labelled inside the panel. No top-N truncation: a curated edge that never cleared `expr_prop`
is a reported negative, listed under `missing_` rather than dropped silently.

```python  # cell 23
import importlib; importlib.reload(U)
f1b = U.plot_dotplot_programs(res, repro, focal=cfg.CD4_MALIGNANT, partners=MAIN_ROSTER,
                              out=U.fig_path("dotplot_programs"))

missing = {k: v for k, v in f1b.missing_.items() if v}
U.printout(
    "FIGURE 1b -- CURATED EDGES BY BIOLOGICAL PROGRAM",
    lines=[
        f"file            : {U.fig_path('dotplot_programs').name}",
        f"programs        : {len(U.CCC_PROGRAMS)}, "
        f"{sum(len(v) for v in U.CCC_PROGRAMS.values())} curated edges",
        "encoding        : as figure 1 (colour = -log10 specificity_rank, size = donor frac); "
        "rows blocked by program, no top-N cut",
        *[f"  not scored in this run -- {k}: {', '.join(v)}" for k, v in missing.items()],
    ],
)
```

```
==============================================================================================================
FIGURE 1b -- CURATED EDGES BY BIOLOGICAL PROGRAM
==============================================================================================================
file            : ccc40_dotplot_programs.svg
programs        : 10, 36 curated edges
encoding        : as figure 1 (colour = -log10 specificity_rank, size = donor frac); rows blocked by program, no top-N cut

<Figure size 1702.25x1922.8 with 3 Axes>
```

### Outcome

`tables/ccc40_reportable_set.csv` is the deliverable; `tables/ccc40_run_log.md` is the
self-describing record of everything above and is readable on its own.

**Deliberately absent.** Stage / disease / layer contrasts (`ccc_data.FORBIDDEN_CONTRASTS` —
each is a study contrast in disguise). Differential abundance. The six roster levels dropped
upstream. `pDC`, from the ranked run only. The doublet arm, which this atlas cannot support.
Spatial co-localisation: li2024's Visium exists and would be the most direct orthogonal check on
any "malignant CD4 signals to F_inflammatory" claim — it is not used here.

**Read with.** Splitting a level into k sub-levels multiplies the tested grid by k while each
sub-level carries fewer cells and fewer donors, so a pair that was solid on pooled `Myeloid` can
look weaker on every sub-level with no biology having changed. The sub-level states were also
clustered on the same expression matrix that is scored here, and two of the resolution tests
(`cxcl12_axis`, `mhcii_axis`) ask which state carries a gene that helped define it — those two
are circular by construction and are reported, not claimed.


