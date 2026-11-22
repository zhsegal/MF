# old/28_tme_degradome_v1.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28_tme_degradome_v1.ipynb` and is never run._

# 28 · The skin degradome — which TME subtype makes the proteases

`24_subclone_tcr_signaling` scores one `protease` program on malignant subclones (`subclone_helpers.py:511`, 11
genes, `sc.tl.score_genes` per cell) and renders it as a single column of
`figures/subclone_v3_programs.png`. It is never tested; the per-gene dot plot is dead code
(`panels` at `24_subclone_tcr_signaling` cell 11 is computed and never consumed); and the one time the panel *was*
tested gene by gene (`old/23_subclonal_evolution.ipynb` cell 17) the sole up-call was
**TIMP1** — the inhibitor. The read "ECM proteolysis in MF skin is not tumour-intrinsic" is
supported by that, and was never followed by the obvious next question.

**This notebook asks who does make it.** `22_myeloid_fibro_reannotation` produced 13 myeloid and fibroblast subtypes
that did not exist when the `24_subclone_tcr_signaling` panel was written; `27_ccc_subtype` uses them for ligand–receptor work
and never for expression. Here every protease, ECM-degrading enzyme and endogenous inhibitor
is attributed to a producing cell type, at that subtype resolution, with donor-level spread.

## What is claimed, and what is only counted

| | levels | role |
|---|---|---|
| **claim** | 13 `22_myeloid_fibro_reannotation` subtypes + `CD4_malignant` | statements are made about these |
| **context** | Keratinocyte, CD4_reactive, CD4_unassessed, CD8, Tregs, B, Plasma, Vascular, Mast, Melanocyte | denominator only — no claim, no test |

"cDC supplies 40% of the skin's MMP12" is meaningless without a complete denominator, so
every skin level is carried. Nothing is asserted about the context levels. **Mast cells will
own several genes; that is the positive control (§2b), not a finding.**

## The power constraint, settled before any code ran

Skin has **9 HC donors**, from 3 studies. At the 25-cell gate the CTCL-vs-HC contrast is
defined for **6 of 13** subtypes — cDC (8 HC donors), F_reticular (8), F_papillary (8),
F_mesenchymal (7), Mac_FOLR2 (5), F_apCAF (4) — and for **none** of DC_LAMP3, Mac_infl,
Mono_moDC, Mac_SPP1_TREM2, LC, pDC, F_inflammatory. Those seven are not null: they are absent
from healthy skin at the gate, which is an abundance observation (§5) and never a fold change.
`chennareddy2025` and `gaydosik2019` each carry both arms, so the contrast runs **within
study**; a pooled fit would put disease and study in the same design column and is kept only
as a labelled sensitivity arm.

## Sections

§0 build gate, roster, coverage · §1 panel coverage · §2 source attribution (+ controls) ·
§3 spillover · §4 CTCL vs HC · §5 the levels with no healthy comparator · §6 figures + log.

Config is `degradome_data.py`, every function is in `degradome_helpers.py`. Nothing is
defined here. The object comes from `jobs/run_degradome_build.sh` — one bsub pass over the
48 GB source atlas, because ADAM10 and ADAM17 are in `24_subclone_tcr_signaling`'s panel and are not among the 10k
HVGs of `joint_mrvi_input_skin.h5ad`.

Every section writes `tables/deg42_*.csv` and appends a self-describing block to
`tables/deg42_run_log.md` that reads correctly pasted into a chat with no access to this
notebook.

**Kernel: `neural_nmf_env`.**

```python  # cell 1
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# The repo is not necessarily an ancestor of cwd (a kernel started from $HOME sees it as a
# descendant), so walking up is a convenience, not the definition. FALLBACK is authoritative
# and matches degradome_data.NB_DIR.
FALLBACK = Path("/home/projects/nyosef/zvise/MF")


def _resolve_nb_dir():
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if cand.name == "MF" and (cand / "data").exists():
            return cand
        if (cand / "notebooks" / "MF" / "data").exists():
            return cand / "notebooks" / "MF"
    if (FALLBACK / "data").exists():
        return FALLBACK
    raise RuntimeError(f"cannot locate the MF tree from {p} and {FALLBACK} is absent")


NB_DIR = _resolve_nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))

# Reloaded explicitly, not merely imported: these two modules are edited while the notebook
# is open, and a kernel that imported them at session start will happily keep serving the old
# code -- which shows up as a TypeError on a parameter that plainly exists on disk.
import importlib                  # noqa: E402
import degradome_data as cfg      # noqa: E402
import degradome_helpers as D     # noqa: E402

cfg = importlib.reload(cfg)
D = importlib.reload(D)
D.cfg = cfg                       # the helper module caches its own reference to the config

# Slow results (the three DESeq2 fits, the shuffle null, the equivalence gate) are cached
# under data/degradome/cache and stamped with the built object's mtime. Set True -- or call
# D.clear_cache() -- to force a recompute; rebuilding the h5ad invalidates them anyway.
RECOMPUTE = False

SEED = cfg.SEED
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"NB_DIR    {NB_DIR}")
print(f"object    {cfg.DEG_ADATA}")
print(f"panel     {len(cfg.panel_genes(False))} degradome + {len(cfg.CONTEXT_GENES)} identity")
print(f"claim     {len(cfg.CLAIM_LEVELS)} levels | context {len(cfg.CONTEXT_LEVELS)}")
```

```
NB_DIR    /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
object    /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/degradome/skin_degradome.h5ad
panel     120 degradome + 36 identity
claim     14 levels | context 10
```

## §0 · The build gate, the roster, the coverage

Three things are established before a single number is read: the panel is complete, the
object's expression values are what the build claims, and the roster reconciles cell-for-cell
against the `22_myeloid_fibro_reannotation` sidecar.

The equivalence gate is **blocking**. `stream_lognorm_subset` sums the library size over all
40,821 genes *before* subsetting to the 156 panel columns. Doing it the other way round
rescales every cell by (panel counts / total counts) — about 40× here — and, worse, by a
factor that varies with cell type, which is exactly the axis this notebook measures. The
result would look entirely plausible and be wrong everywhere. The test set is 3 donors at
full gene width; every panel value must match to `<1e-6`.

```python  # cell 3
adata = D.load_degradome_adata()

# The gate reads a 95,233 x 40,821 test set, so its verdict is cached rather than its data.
gate = D.cached("equivalence",
                lambda: {"max_abs_diff": D.assert_build_equivalence(adata)},
                force=RECOMPUTE)
print(f"equivalence gate: max abs diff {gate['max_abs_diff']:.3e}")
```

```
roster reconciled: 13 nb10c levels, 153,164 cells, zero drift
skin_degradome.h5ad: 720,125 cells x 156 genes, 82 donors, 99 samples
  build dropped 29,385 of 749,510 skin cells with no identity
  DEGRADOME-ONLY object: var is a hand-curated 156-gene panel. Not for HVG selection, PCA, UMAP, clustering or genome-wide DE. Use skin_degradome_pseudobulk_full.parquet for anything that needs all genes.
[cache] equivalence <- equivalence.json
equivalence gate: max abs diff 0.000e+00
```

```python  # cell 4
cov = D.coverage_table(adata.obs, out=cfg.tab("coverage"))
testable = D.assert_power_statement(cov)
```

```
degradome_level  claim  n_cells  n_cells_ctcl  n_cells_hc  n_donors_present  n_donors_ge25  n_hc_donors_ge25  n_studies       top_study  top_study_frac  claimable  study_dominated  disease_testable
             LC   True     2861          2846          15                69             15                 0          7          li2024           0.749       True            False             False
            cDC   True    30650         29952         698                73             65                 8          7          li2024           0.427       True            False              True
       DC_LAMP3   True    19068         19061           7                71             45                 0          7          li2024           0.882       True             True             False
            pDC   True     1113          1111           2                64             13                 0          7          li2024           0.608       True            False             False
      Mono_moDC   True     7047          7017          30                72             43                 0          7 chennareddy2025           0.547       True            False             False
      Mac_FOLR2   True    13943         13543         400                73             55                 5          7 chennareddy2025           0.481       True            False              True
 Mac_SPP1_TREM2   True     3835          3809          26                67             21                 0          7          li2024           0.783       True            False             False
       Mac_infl   True    12731         12720          11                66             31                 0          7          li2024           0.617       True            False             False
    F_papillary   True     8262          7006        1256                71             36                 8          7          li2024           0.346       True            False              True
    F_reticular   True    23128         18522        4606                69             42                 8          7 chennareddy2025           0.311       True            False              True
  F_mesenchymal   True     6146          4933        1213                60             34                 7          7          li2024           0.529       True            False              True
 F_inflammatory   True    14144         14131          13                51             11                 0          6          li2024           0.930       True             True             False
        F_apCAF   True    10236         10043         193                69             42                 4          7 chennareddy2025           0.474       True            False              True
  CD4_malignant   True    76349         76349           0                31             29                 0          6          li2024           0.570       True            False             False
   CD4_reactive  False    72207         71488         719                33             32                 2          6          li2024           0.747       True            False             False
 CD4_unassessed  False   190142        189337         805                73             71                 8          7          li2024           0.548       True            False              True
            CD8  False    51831         51531         300                72             63                 2          7          li2024           0.689       True            False             False
          Tregs  False     1936          1931           5                66             20                 0          6          li2024           0.411       True            False             False
              B  False    19388         19377          11                65             41                 0          7          li2024           0.742       True            False             False
         Plasma  False    10511         
... [1202 more chars truncated]
```

## §1 · Panel coverage

Detection fraction per (gene, level) over eligible donors. Genes never detected above 1% of
cells in any level are dropped from the attribution — a share computed from three counts is
noise, not attribution — and the drop list is printed, because silent truncation reads as
"we looked at everything" when we did not.

The `(level, donor)` cube is built here by one indicator matmul. **Donor, not sample, is the
replicate unit**: several biopsies from one patient are not independent observations of that
patient's stroma.

```python  # cell 6
cube = D.DegradomeCube(adata)
panel_cov, dropped = D.panel_coverage(cube, out=cfg.tab("panel_coverage"))
GENES = [g for g in cfg.panel_genes(include_context=False) if g in set(panel_cov.index[panel_cov["keep"]])]
print(f"\n{len(GENES)} degradome genes carried into the attribution")
```

```
cube (24, 82, 156) | 982 eligible (level, donor) pairs at >= 25 cells
panel: 148 of 156 genes detected in >= 1% of cells in at least one level
  dropped (8): CTSE, KLK12, KLK14, KLK9, MMP24, MMP8, PRTN3, TMPRSS11D

112 degradome genes carried into the attribution
```

## §2 · Source attribution — the primary claim

Per donor, per gene: what fraction of that donor's total skin transcript output for the gene
came from each level. Two shares, answering different questions.

- **`share_raw`** — of every transcript of gene *g* captured in this donor's skin, how much
  came from level *l*. Abundance is part of the answer: a fibroblast population ten times
  larger genuinely supplies ten times the collagenase. **This is the attribution.**
- **`share_cpm`** — the same after normalising each level by its own full-gene library, so
  every level contributes per unit of transcriptome. Abundance divided out; this is intensity.

A gene owned on `share_raw` but not on `share_cpm` is supplied by a level because there is a
lot of it, not because each cell makes much. Both are reported.

A level **owns** a gene when it is the top producer in ≥⅔ of the donors where the share is
computable, over ≥5 donors. Genes clearing neither get `owner = None` — shared production is
a real answer, not a missing one.

```python  # cell 8
attr = D.attribution(cube, genes=GENES, out=cfg.tab("attribution"))
own = D.call_owners(attr, out=cfg.tab("gene_owner"))
```

```
22 of 112 genes have a single-level owner at 67% of donors
owner
Keratinocyte      18
Vascular           2
F_reticular        1
CD4_unassessed     1
```

### §2b · Controls

**Positive.** The attribution has a known right answer for a handful of genes: mast tryptases
must come from mast cells, granzymes from CD8, the kallikrein cascade and SPINK5 from
keratinocytes, COL1A1 from fibroblasts. These are not results — they are the only way to know
the share arithmetic, the `cell_id` join and the library sizes are right before any actual
claim is read. A miss hard-fails.

**Negative.** The level labels are permuted within donor, which preserves every level's size
and every donor's composition and destroys only the cell-to-level link. The resulting win
fraction is what this many donors and this much sparsity produce by chance. An owner call
that does not clear that ceiling is not an attribution.

```python  # cell 10
ctrl = D.cached("positive_controls",
                lambda: D.assert_positive_controls(adata, strict=False),
                force=RECOMPUTE)
D.assert_controls_pass(ctrl)      # gated even when cached
ctrl.to_csv(cfg.tab("positive_controls"), index=False)
```

```
[cache] positive_controls <- positive_controls.parquet
statistic      tier   gene              expected    top_level        owner  win_frac  n_donors  pass note
      raw ownership   KLK5        [Keratinocyte] Keratinocyte Keratinocyte     1.000        52  True     
      raw ownership   KLK7        [Keratinocyte] Keratinocyte Keratinocyte     1.000        55  True     
      raw ownership SPINK5        [Keratinocyte] Keratinocyte Keratinocyte     0.957        70  True     
      raw ownership COL1A1          [Fibroblast]   Fibroblast   Fibroblast     0.975        81  True     
      raw ownership COL1A2          [Fibroblast]   Fibroblast   Fibroblast     0.975        81  True     
      raw ownership   C1QA             [Myeloid]      Myeloid      Myeloid     0.976        82  True     
      raw ownership   CTSS             [Myeloid]      Myeloid      Myeloid     0.841        82  True     
      raw ownership   CTSK [Fibroblast, Myeloid]   Fibroblast   Fibroblast     0.914        81  True     
      cpm ownership   GZMA                 [CD8]          CD8          CD8     0.827        81  True     
      cpm ownership   GZMK                 [CD8]          CD8          CD8     0.987        78  True     
      cpm ownership   GZMH                 [CD8]          CD8          CD8     0.948        77  True     
      cpm ownership  CD207             [Myeloid]      Myeloid      Myeloid     0.929        70  True     
      cpm ownership  MLANA          [Melanocyte]   Melanocyte   Melanocyte     0.894        66  True     
      raw plurality TPSAB1                [Mast]         Mast         None     0.361        36  True     
      raw plurality  TPSB2                [Mast]         Mast         None     0.394        33  True     
      raw plurality   CPA3                [Mast]         Mast         None     0.569        51  True     
      cpm plurality   CMA1                [Mast]         Mast         None     0.636        33  True     
      cpm plurality TPSAB1                [Mast]         Mast         None     0.417        36  True     

18/18 positive controls recovered
```

```python  # cell 11
null = D.cached("shuffle_null",
                lambda: D.shuffle_null(adata, cube, genes=GENES),
                force=RECOMPUTE)
own = D.against_null(own, null, out=cfg.tab("gene_owner"))
```

```
[cache] shuffle_null <- shuffle_null.parquet
22 of 22 owner calls exceed the within-donor shuffle ceiling; 0 do not and are not attributions
```

### §2c · The answer to `24_subclone_tcr_signaling`'s question

The same arithmetic, restricted to the twelve genes `24_subclone_tcr_signaling` actually scored. If the tumour clone
is not the source, this table names what is.

```python  # cell 13
D.assert_nb31_panel()

# (a) subtype roster -- "which macrophage state", the resolution nb10c bought us
nb31 = own[own["gene"].isin(cfg.NB31_PANEL)].set_index("gene").reindex(cfg.NB31_PANEL)
mal = attr[(attr[cfg.GROUPBY] == cfg.CD4_MALIGNANT) & (attr["gene"].isin(cfg.NB31_PANEL))]
nb31["CD4_malignant_share"] = mal.set_index("gene")["share_raw_med"].reindex(cfg.NB31_PANEL).round(4)
nb31 = nb31.reset_index()
nb31.to_csv(cfg.tab("nb31_panel_owner_subtype"), index=False)
print("=== subtype roster ===")
print(nb31[["gene", "family", "owner", "owner_win_frac", "owner_share_med",
            "CD4_malignant_share", "n_donors_abstained"]].to_string(index=False))

# (b) lineage roster -- the granularity that can answer "tumour or stroma". A gene with no
# single-subtype owner is not a gene with no source; production split across five fibroblast
# states still has a lineage.
attr_lin, own_lin, cube_lin = D.lineage_attribution(adata, cfg.NB31_PANEL, verbose=False)
mal_lin = attr_lin[attr_lin[cfg.GROUPBY] == cfg.CD4_MALIGNANT]
lin = own_lin.set_index("gene").reindex(cfg.NB31_PANEL)
lin["CD4_malignant_share"] = mal_lin.set_index("gene")["share_raw_med"].reindex(cfg.NB31_PANEL).round(4)
lin = lin.reset_index()
lin.to_csv(cfg.tab("nb31_panel_owner_lineage"), index=False)
print("\n=== lineage roster (TME subtypes collapsed to Myeloid / Fibroblast) ===")
print(lin[["gene", "owner", "owner_win_frac", "owner_share_med",
           "CD4_malignant_share"]].to_string(index=False))

n_mal = int((lin["owner"] == cfg.CD4_MALIGNANT).sum())
print(f"\nof {len(lin)} nb31 panel genes, {n_mal} are owned by CD4_malignant at lineage "
      f"granularity; median malignant share = {lin['CD4_malignant_share'].median():.4f}")
```

```
=== subtype roster ===
   gene        family owner  owner_win_frac  owner_share_med  CD4_malignant_share  n_donors_abstained
  MMP14           mmp  None           0.235           0.1346               0.0011                   6
  MMP25           mmp  None           0.378           0.1477               0.0939                   4
   MMP9           mmp  None           0.354           0.3345               0.0052                  10
   MMP2           mmp  None           0.395           0.5672               0.0001                  15
   PLAU  serine_other  None           0.438           0.3936               0.0022                   4
  PLAUR  serine_other  None           0.500           0.3392               0.0066                   4
   GZMB      granzyme  None           0.395           0.3228               0.0640                   5
 ADAM17 metallo_other  None           0.346           0.2174               0.1506                   0
 ADAM10 metallo_other  None           0.366           0.2138               0.2143                   1
ADAMTS4 metallo_other  None           0.575           0.4337               0.0011                   5
   CTSC     cathepsin  None           0.280           0.1182               0.1320                   2
  TIMP1     inhibitor  None           0.195           0.0433               0.0519                   4

=== lineage roster (TME subtypes collapsed to Myeloid / Fibroblast) ===
   gene      owner  owner_win_frac  owner_share_med  CD4_malignant_share
  MMP14       None           0.568           0.4675               0.0011
  MMP25       None           0.581           0.4663               0.0939
   MMP9    Myeloid           0.963           0.9232               0.0052
   MMP2 Fibroblast           0.827           0.9234               0.0001
   PLAU       None           0.438           0.3333               0.0022
  PLAUR    Myeloid           0.756           0.6993               0.0066
   GZMB       None           0.395           0.3228               0.0640
 ADAM17       None           0.309           0.1131               0.1506
 ADAM10       None           0.354           0.2138               0.2143
ADAMTS4       None           0.487           0.4337               0.0011
   CTSC       None           0.439           0.2848               0.1320
  TIMP1       None           0.476           0.4300               0.0519

of 12 nb31 panel genes, 0 are owned by CD4_malignant at lineage granularity; median malignant share = 0.0293
```

## §3 · Spillover control

Ambient mRNA and doublets put a cell type's transcripts into every other cell type in the same
droplet suspension, in proportion to how abundant that cell type is. So for a level that does
**not** own a gene, a per-cell intensity (abundance already divided out) that still rises with
the owner's donor cell fraction is contamination, not expression.

Flagged, never dropped: an inflamed donor really does have both more macrophages and more
macrophage-adjacent signalling, so a positive correlation is suggestive and not proof. The
second guard is detection — ambient raises the mean at low detection, so a high share with a
low `det_frac` is the shape to distrust.

```python  # cell 15
spill = D.spillover_flags(cube, own, genes=GENES, out=cfg.tab("spillover"))
own = D.owner_spillover_note(own, cube)
own.to_csv(cfg.tab("gene_owner"), index=False)
low = own[own["owner"].notna() & own["owner_low_detection"]]
print(f"\n{len(low)} owner calls rest on a high share at < 10% detection in the owner:")
print(low[["gene", "owner", "owner_share_med", "owner_det_frac"]].to_string(index=False))
```

```
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/degradome_helpers.py:937: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  "gene": g, "family": cfg.GENE_TO_FAMILY.get(g, "context"),

20 of 504 (gene, non-owner level) pairs track the owner's abundance (rho >= 0.5, p < 0.05) -- flagged, not dropped
   gene degradome_level        owner  rho_vs_owner_abundance  det_frac_med
  MMP13  F_inflammatory Keratinocyte                   0.880         0.000
  MMP13   F_mesenchymal Keratinocyte                   0.733         0.000
  MMP28  F_inflammatory Keratinocyte                   0.645         0.002
  KLK11              LC Keratinocyte                   0.632         0.000
   CSTA        Vascular Keratinocyte                   0.604         0.008
   KLK8           Tregs Keratinocyte                   0.570         0.000
  KLK11     F_reticular Keratinocyte                   0.568         0.000
ADAMTS9   F_mesenchymal     Vascular                   0.554         0.063
   KLK8              LC Keratinocyte                   0.554         0.000
   KLK8        DC_LAMP3 Keratinocyte                   0.538         0.000

9 owner calls rest on a high share at < 10% detection in the owner:
     gene          owner  owner_share_med  owner_det_frac
     CTSV   Keratinocyte           0.8378           0.060
      PI3   Keratinocyte           0.9288           0.046
    MMP13   Keratinocyte           0.8167           0.000
    ELANE    F_reticular           0.7824           0.070
    KLK13   Keratinocyte           0.9487           0.041
     KLK5   Keratinocyte           0.9649           0.068
     KLK6   Keratinocyte           0.9919           0.006
TMPRSS11E CD4_unassessed           0.7411           0.002
 TMPRSS13   Keratinocyte           0.8537           0.077
```

### §3b · Protease : inhibitor balance

The readout the `24_subclone_tcr_signaling` comment promised — *"read the protease:TIMP1 ratio in the dot plot"* —
and never rendered, computed on families rather than one inhibitor, in CPM on each level's own
full-gene library, so it is net degradative tone per unit of transcriptome and does not move
with cell number.

```python  # cell 17
bal = D.balance_table(cube, out=cfg.tab("balance"))
```

```
balance          cathepsin_cystatin  mmp_timp  serine_serpin
degradome_level                                             
LC                            -1.60      0.08          -0.05
cDC                           -2.27     -0.06          -0.43
DC_LAMP3                      -1.09      0.72          -2.58
pDC                           -3.14     -2.93          -0.81
Mono_moDC                     -0.79     -0.92          -0.22
Mac_FOLR2                     -0.20      1.46          -0.15
Mac_SPP1_TREM2                 1.01      2.63          -0.43
Mac_infl                      -0.25      0.30          -0.89
F_papillary                    0.34     -0.12          -1.22
F_reticular                   -0.76     -0.70          -1.03
F_mesenchymal                  0.04     -0.93          -0.21
F_inflammatory                 0.22     -0.12          -2.00
F_apCAF                       -0.20     -1.45          -1.46
CD4_malignant                  0.24     -3.05          -1.01
CD4_reactive                   0.68     -2.94          -1.05
CD4_unassessed                 0.74     -2.01          -0.91
CD8                           -0.13     -1.25          -2.13
Tregs                          0.85     -1.68          -2.65
B                              1.87     -1.91          -3.00
Plasma                         1.04     -0.52          -0.70
Keratinocyte                  -1.08      0.56          -0.33
Vascular                      -0.42     -2.51          -0.53
Mast                           0.85     -4.95          -0.37
Melanocyte                     0.47     -1.79          -0.58
```

## §4 · CTCL vs healthy skin, within level

Pseudobulk DESeq2 on the **full-gene** cube (so size factors and dispersions are estimated on
the whole transcriptome, not on 156 columns), collapsed to donor, run separately inside each
level, design `~ study + disease_grp`. BH is applied within level across the panel genes,
because the panel is the hypothesis.

Restricted to `chennareddy2025` + `gaydosik2019` — the only two skin studies contributing
donors to both arms. Five of seven are CTCL-only, so a pooled fit reports batch as biology;
it is run below as a labelled sensitivity arm and is not a headline.

```python  # cell 19
agg, dmeta = D.donor_pseudobulk()
de = D.cached("contrast_disease",
              lambda: D.disease_contrast(agg, dmeta, levels=testable, arm="within_study"),
              force=RECOMPUTE)
de_pooled = D.cached("contrast_disease_pooled",
                     lambda: D.disease_contrast(agg, dmeta, levels=testable,
                                                arm="pooled_sensitivity"),
                     force=RECOMPUTE)

# epidermis vs dermis: paired within donor, so the pseudobulk must keep the layer split.
# Needs no healthy arm, which is why it reaches DC_LAMP3 and Mac_infl.
de_layer = D.cached("contrast_layer",
                    lambda: D.layer_contrast(*D.donor_pseudobulk(extra_keys=(cfg.LAYER_KEY,))),
                    force=RECOMPUTE)

# advanced vs early: li2024 is the only study containing both stages, so the contrast is run
# inside it and is free of study confounding by construction.
de_stage = D.cached("contrast_stage", lambda: D.stage_contrast(agg, dmeta), force=RECOMPUTE)

de_all = pd.concat([de, de_pooled, de_layer, de_stage], ignore_index=True)
de_all.to_csv(cfg.tab("contrasts"), index=False)

sig = de[(de["padj_panel"] < 0.05) & (de["log2FoldChange"].abs() >= 1.0)]
print(f"\nCTCL vs HC within study: {len(sig)} calls at panel FDR < 0.05, |log2FC| >= 1")
print(sig.sort_values("padj_panel")[["gene", "family", cfg.GROUPBY, "log2FoldChange",
                                     "padj_panel"]].head(20).to_string(index=False))
for name, d in [("epidermis vs dermis", de_layer), ("advanced vs early", de_stage)]:
    s = d[(d["padj_panel"] < 0.05) & (d["log2FoldChange"].abs() >= 1.0)] if len(d) else d
    print(f"\n{name}: {len(s)} calls; levels {sorted(set(d[cfg.GROUPBY])) if len(d) else '(none)'}")
    if len(s):
        print(s.sort_values("padj_panel")[["gene", "family", cfg.GROUPBY, "log2FoldChange",
                                           "padj_panel"]].head(15).to_string(index=False))
```

```
pseudobulk collapsed to level x donor: 1,685 rows x 40,821 genes
[cache] contrast_disease <- contrast_disease.parquet
[cache] contrast_disease_pooled <- contrast_disease_pooled.parquet
[cache] contrast_layer <- contrast_layer.parquet
[cache] contrast_stage <- contrast_stage.parquet

CTCL vs HC within study: 127 calls at panel FDR < 0.05, |log2FC| >= 1
    gene        family degradome_level  log2FoldChange   padj_panel
     MME metallo_other     F_reticular        2.773353 1.188191e-11
   ELANE  serine_other     F_papillary       -3.783404 2.528215e-09
    CTSS     cathepsin     F_papillary        1.940863 3.228273e-09
  PRSS23  serine_other     F_reticular        3.073444 3.654132e-09
    MMP1           mmp     F_papillary       10.224161 1.286487e-08
 ADAMTS5 metallo_other     F_papillary        2.712075 3.797611e-08
   MMP12           mmp             cDC        3.878184 3.875734e-08
    CTSA     cathepsin     F_papillary        1.227398 4.717704e-08
  ADAM12 metallo_other     F_reticular        4.754157 1.238338e-07
ADAMDEC1 metallo_other     F_papillary        7.157538 3.094828e-07
    CTSC     cathepsin     F_papillary        2.276523 6.132868e-07
    CTSC     cathepsin   F_mesenchymal        3.598550 8.663003e-07
    PLAT  serine_other             cDC       -3.393627 1.490807e-06
SERPINE1     inhibitor     F_reticular        3.166525 2.451896e-06
   PLAUR  serine_other             cDC       -1.509071 2.676539e-06
  PRSS23  serine_other     F_papillary        3.362915 2.856211e-06
     MME metallo_other     F_papillary        3.656861 6.931639e-06
    PLAU  serine_other     F_reticular        2.244563 3.221813e-05
  ADAM10 metallo_other     F_papillary        1.128537 4.355577e-05
   TIMP3     inhibitor     F_papillary       -2.622706 6.660284e-05

epidermis vs dermis: 0 calls; levels (none)

advanced vs early: 13 calls; levels ['DC_LAMP3', 'F_apCAF', 'F_mesenchymal', 'F_papillary', 'F_reticular', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_moDC', 'cDC']
  gene        family degradome_level  log2FoldChange  padj_panel
PRSS23  serine_other     F_reticular        1.772338    0.000362
  MMP1           mmp         F_apCAF       -4.570206    0.005095
 MMP19           mmp   F_mesenchymal        1.487832    0.014416
  MMP9           mmp   F_mesenchymal        4.246561    0.014416
  CTSB     cathepsin     F_reticular        1.571462    0.016448
ADAM12 metallo_other     F_reticular        2.095974    0.022547
 TIMP3     inhibitor   F_mesenchymal        2.154910    0.029613
 ELANE  serine_other   F_mesenchymal       -3.272553    0.029613
  PLAT  serine_other   F_mesenchymal        2.362313    0.029625
  GZMA      granzyme         F_apCAF        2.452416    0.031179
 HTRA1  serine_other         F_apCAF        1.286153    0.031179
 HTRA3  serine_other     F_reticular        1.918886    0.038476
   MME metallo_other         F_apCAF        1.764264    0.038927
```

```python  # cell 20
# Agreement between the two arms -- a within-study call that flips sign when pooled is a
# study effect wearing a disease label.
if len(de) and len(de_pooled):
    j = de.merge(de_pooled, on=["gene", cfg.GROUPBY], suffixes=("_within", "_pooled"))
    j["sign_agrees"] = np.sign(j["log2FoldChange_within"]) == np.sign(j["log2FoldChange_pooled"])
    s = j[j["padj_panel_within"] < 0.05]
    print(f"of {len(s)} within-study calls, {int(s['sign_agrees'].sum())} keep their sign when "
          f"every study is pooled")
```

```
of 154 within-study calls, 154 keep their sign when every study is pooled
```

## §5 · The seven levels with no healthy comparator

DC_LAMP3, Mac_infl, Mono_moDC, Mac_SPP1_TREM2, LC, pDC and F_inflammatory clear the claim gate
in lesional skin and have **zero** HC donors at that gate. Their degradome output is reported
here as an abundance observation with no fold change attached. Writing them as "not
significant" would be the single most misleading thing this notebook could do.

```python  # cell 22
lesion = D.lesion_restricted_table(cov, cube, out=cfg.tab("lesion_restricted"))
```

```
levels present in lesional skin with no healthy comparator (abundance observation, NOT a fold change):
degradome_level  n_cells_ctcl  n_donors_ge25_ctcl  n_cells_hc  n_hc_donors_ge25                                                contrast  mmp_cpm_med  cathepsin_cpm_med  granzyme_cpm_med  serine_other_cpm_med  metallo_other_cpm_med  inhibitor_cpm_med
       DC_LAMP3         19061                  45           7                 0 undefined -- no healthy comparator at the coverage gate   265.786583        1806.500557         79.977606            170.782954             612.884723        5860.891735
       Mac_infl         12720                  31          11                 0 undefined -- no healthy comparator at the coverage gate  1154.304438        6250.766169         34.073872            910.019394             511.323033       11133.641707
      Mono_moDC          7017                  43          30                 0 undefined -- no healthy comparator at the coverage gate   854.736147        3576.461115         48.383715           1217.832350             514.672178       10735.739962
 Mac_SPP1_TREM2          3809                  21          26                 0 undefined -- no healthy comparator at the coverage gate  3567.615965       12443.253523         10.930375            876.251756             873.737836        8765.087846
             LC          2846                  15          15                 0 undefined -- no healthy comparator at the coverage gate  1077.481966        2729.647696          8.501580            697.050561             558.478945       11533.029731
            pDC          1111                  13           2                 0 undefined -- no healthy comparator at the coverage gate   134.935036        2092.663942        276.903346            447.462494             419.146170       19484.287182
 F_inflammatory         14131                  11          13                 0 undefined -- no healthy comparator at the coverage gate  3468.701932        3941.951306          4.113793           1000.856804            1309.439940       13191.378226
```

## §6 · Figures

Four figures, one question each: who supplies the `24_subclone_tcr_signaling` panel, the protease:inhibitor balance
per level, and the three contrasts. The gene x level heatmap over the whole panel was dropped
-- 112 rows against 24 columns is a lookup table, and `tables/deg42_attribution.csv` is the
lookup table.

```python  # cell 24
D.plot_nb31_panel_owner(attr, save=cfg.fig("nb31_panel_owner"))
```

```
<Figure size 825x618.2 with 1 Axes>
<Figure size 825x618.2 with 1 Axes>
```

```python  # cell 25
D.plot_balance(bal, save=cfg.fig("protease_inhibitor_balance"))
```

```
<Figure size 1386x800.8 with 3 Axes>
<Figure size 1386x800.8 with 3 Axes>
```

```python  # cell 26
D.plot_volcano(de, levels=cfg.DISEASE_TESTABLE, arm="within_study",
               title="CTCL vs healthy skin", save=cfg.fig("volcano_disease"))
D.plot_volcano(de_layer, title="epidermis vs dermis", save=cfg.fig("volcano_layer"))
D.plot_volcano(de_stage, title=f"advanced vs early ({cfg.STAGE_STUDY})",
               save=cfg.fig("volcano_stage"))
```

```
<Figure size 1122x660 with 6 Axes>
nothing to plot

<Figure size 1122x1320 with 12 Axes>
<Figure size 1122x1320 with 12 Axes>
```

### §6b · The machine-readable summary

One block per section, appended to `tables/deg42_run_log.md`. It has to be readable pasted
into a chat by someone with no access to this notebook, so every number carries its
denominator and every abstention says why.

```python  # cell 28
called = own[own["owner"].notna()]
mal_owned = called[called["owner"] == cfg.CD4_MALIGNANT]


def _contrast_line(name, d, extra=""):
    if not len(d):
        return f"{name:<26}: no level cleared the design {extra}"
    s = d[(d["padj_panel"] < 0.05) & (d["log2FoldChange"].abs() >= 1.0)]
    return (f"{name:<26}: {len(s)} calls at panel FDR < 0.05, |log2FC| >= 1 across "
            f"{sorted(set(d[cfg.GROUPBY]))} {extra}")


body = "\n".join([
    f"object          : {cfg.DEG_ADATA.name}  {adata.n_obs:,} cells x {adata.n_vars} genes",
    f"donors / samples: {adata.obs[cfg.DONOR_KEY].nunique()} / {adata.obs[cfg.SAMPLE_KEY].nunique()}",
    f"levels          : {len(cfg.CLAIM_LEVELS)} claim, {len(cfg.CONTEXT_LEVELS)} context (denominator only)",
    f"panel           : {len(GENES)} of {len(cfg.panel_genes(False))} degradome genes cleared detection",
    f"dropped genes   : {', '.join(dropped) if dropped else '(none)'}",
    "",
    f"genes with a single-SUBTYPE owner : {len(called)} of {len(own)}",
    f"  owned by CD4_malignant          : {len(mal_owned)}"
    + (f"  ({', '.join(mal_owned['gene'])})" if len(mal_owned) else ""),
    f"  above the shuffle ceiling       : {int((called['above_null'] == True).sum())}",
    f"  resting on < 10% detection      : {int(called['owner_low_detection'].sum())}",
    "",
    "nb31 protease panel at LINEAGE granularity (the resolution that can say tumour vs stroma):",
    lin[["gene", "owner", "owner_win_frac", "owner_share_med",
         "CD4_malignant_share"]].to_string(index=False),
    "",
    f"spillover-flagged (gene, non-owner level) pairs: "
    f"{int(spill['spillover_suspect'].sum()) if len(spill) else 0} of {len(spill)}",
    "",
    "CONTRASTS",
    _contrast_line("CTCL vs HC", de, f"[within {' + '.join(cfg.PAIRED_STUDIES)}]"),
    _contrast_line("epidermis vs dermis", de_layer, "[paired within donor]"),
    _contrast_line("advanced vs early", de_stage, f"[within {cfg.STAGE_STUDY}]"),
    "",
    f"no healthy comparator at the gate (abstained, NOT null): {cfg.DISEASE_UNDEFINED}",
    f"  of these, testable across layers instead: "
    f"{[l for l in cfg.LAYER_TESTABLE if l in cfg.DISEASE_UNDEFINED]}",
])
D.append_run_log("SECTIONS 0-5 -- DEGRADOME ATTRIBUTION", body)
```

```
==============================================================================================================
SECTIONS 0-5 -- DEGRADOME ATTRIBUTION
==============================================================================================================
object          : skin_degradome.h5ad  720,125 cells x 156 genes
donors / samples: 82 / 99
levels          : 14 claim, 10 context (denominator only)
panel           : 112 of 120 degradome genes cleared detection
dropped genes   : CTSE, KLK12, KLK14, KLK9, MMP24, MMP8, PRTN3, TMPRSS11D

genes with a single-SUBTYPE owner : 22 of 112
  owned by CD4_malignant          : 0
  above the shuffle ceiling       : 22
  resting on < 10% detection      : 9

nb31 protease panel at LINEAGE granularity (the resolution that can say tumour vs stroma):
   gene      owner  owner_win_frac  owner_share_med  CD4_malignant_share
  MMP14       None           0.568           0.4675               0.0011
  MMP25       None           0.581           0.4663               0.0939
   MMP9    Myeloid           0.963           0.9232               0.0052
   MMP2 Fibroblast           0.827           0.9234               0.0001
   PLAU       None           0.438           0.3333               0.0022
  PLAUR    Myeloid           0.756           0.6993               0.0066
   GZMB       None           0.395           0.3228               0.0640
 ADAM17       None           0.309           0.1131               0.1506
 ADAM10       None           0.354           0.2138               0.2143
ADAMTS4       None           0.487           0.4337               0.0011
   CTSC       None           0.439           0.2848               0.1320
  TIMP1       None           0.476           0.4300               0.0519

spillover-flagged (gene, non-owner level) pairs: 20 of 504

CONTRASTS
CTCL vs HC                : 127 calls at panel FDR < 0.05, |log2FC| >= 1 across ['F_apCAF', 'F_mesenchymal', 'F_papillary', 'F_reticular', 'Mac_FOLR2', 'cDC'] [within chennareddy2025 + gaydosik2019]
epidermis vs dermis       : no level cleared the design [paired within donor]
advanced vs early         : 13 calls at panel FDR < 0.05, |log2FC| >= 1 across ['DC_LAMP3', 'F_apCAF', 'F_mesenchymal', 'F_papillary', 'F_reticular', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_moDC', 'cDC'] [within li2024]

no healthy comparator at the gate (abstained, NOT null): ['DC_LAMP3', 'Mac_infl', 'Mono_moDC', 'Mac_SPP1_TREM2', 'LC', 'pDC', 'F_inflammatory']
  of these, testable across layers instead: ['DC_LAMP3', 'Mac_infl']

PosixPath('/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/tables/deg42_run_log.md')
```

### Outcome

Fill this in from the printed summary once the notebook has run clean: which subtypes own the
MMP, cathepsin and serine arms; whether `CD4_malignant` owns anything at all; and which of the
six testable levels induce their degradome in lesional skin. State the abstentions in the same
breath — seven subtypes have no healthy comparator, and that is a coverage fact about this
atlas, not a biological null.
