# old/ccc_v1_02_descriptive.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/ccc_v1_02_descriptive.ipynb` and is never run._

# 35 · CCC — descriptive ligand–receptor map, malignant CD4 ↔ TME

Runs off `data/ccc/ccc_skin.h5ad` (built and gated in **nb34**), ~727k skin cells × ~1,832
consensus-resource genes. Interactive; no LSF job needed.

This notebook is the **exploratory record** — every result frame lands in `tables/ccc_*.csv`.
The figures for reading are re-derived from scratch in **nb36**.

## Design

- Malignancy is **ALICE-TCR alone** (`mal_tcr_alice`), not `mal_combined`. The call is a
  TCR-sequence fact, so it is independent of the expression LIANA scores; inferCNV's is not.
  §13 keeps `mal_combined` / `mal_cnv` as sensitivity arms.
- Senders: `CD4_malignant` (76,349 cells, 29 donors) **and** `CD4_reactive` (72,207, 34 donors).
  The reactive comparator is what makes "malignant-specific" mean anything — it is the same
  lineage in the same lesion, so donor, study, chemistry and ambient RNA all affect it too.
  40,447 of the reactive cells do carry a `cnv_only` call; §13 tests that disagreement rather
  than assuming it away.
- Partners: CD8 · Myeloid · Fibroblast (core, 24–28 donors) then Keratinocyte · Vascular · B ·
  Plasma · Tregs (extended, 10–26 donors). Mast / Melanocyte are run but never claimed.
- Pooled over all 7 studies and all CTCL disease labels. **No stage, disease or layer
  stratification** — see `ccc_data.FORBIDDEN_CONTRASTS`: early-vs-advanced exists only inside
  li2024 (7 v 7 donors) and SS skin is buus2025 alone.

## How to read the numbers in here

LIANA's permutation p-value takes the **cell** as its unit. 76,349 malignant CD4 cells come from
29 donors, so those p-values are pseudoreplicated by roughly 2,600× and will approach 0 for
almost any pair. They size dots and mark cells; **they are not evidence**. What carries weight
here is: (i) the rank *delta* against the reactive comparator, (ii) stability across `expr_prop`,
(iii) survival of a within-donor label shuffle, and (iv) stability across the four malignancy
definitions. Donor-level inference is the deferred differential phase.

**Sections**: §1 load + coverage · §2 core axis · §3 extended · §4 thin · §5 malignant↔reactive ·
§6 comparator + rank delta · §7 controls · §8 expr_prop sweep · §9 within-donor shuffle ·
§10 downsample stability · §11 forced curated panel · §12 study diagnostic ·
§13 malignancy-definition sensitivity · §14 CD8 cnv-call sensitivity · §15 HC negative control.

```python  # cell 1
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
import matplotlib as mpl, matplotlib.pyplot as plt
import plotnine as p9
import liana as li


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); sys.path.insert(0, str(NB_DIR))
import ccc_data as cd
import ccc_helpers as C

sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
cd.TAB_DIR.mkdir(parents=True, exist_ok=True); cd.FIG_DIR.mkdir(parents=True, exist_ok=True)
print("liana", li.__version__)
print("\n" + cd.CAVEAT_BLOCK)
```

```
liana 1.8.1

LIANA permutation p-values are computed with the CELL as the unit and are pseudoreplicated across donors (76,349 malignant CD4 cells come from 29 donors). They size the dots and mark the heatmap cells; they are not evidence. Donor-level inference is the deferred differential phase. Grey heatmap cells mean the pair failed min_cells in that stratum -- read them against the coverage table printed beneath each figure, not as biology.
```

## §1 · Load, resource, coverage, balanced subsample

```python  # cell 3
adata = C.load_ccc_adata()
adata.layers[cd.LAYER] = adata.X          # liana reads layer=LAYER; X already is lognorm
C.assert_ccc_invariants(adata)
resource, coverage = C.load_resource(var_names=adata.var_names)
```

```
AnnData object with n_obs × n_vars = 727156 × 1832
    obs: 'donor', 'real_donor', 'sample_id', 'study', 'dataset', 'compartment', 'cell_type_final', 'cell_type_broad', 'cell_type', 'cell_type_T2', 'lineage', 'mal_combined', 'mal_cnv', 'mal_tcr_alice', 'mal_tcr_dom', 'malignant_evidence', 'assessed_cnv', 'assessed_tcr', 'assessed_tcr_nb30', 'cnv_score', 'clone_id', 'clone_size', 'is_expanded', 'disease', 'disease_stage', 'entity', 'stage_class', 'stage_clean', 'stage_group', 'skin_layer', 'tissue', 'lesion_type', 'treatment_context', 'blood_involvement', 'sex', 'tech', 'cohort_region', 'n_genes', 'total_counts', 'pct_mito', 'doublet_score', 'ccc_celltype'
    uns: 'ccc_build'
    layers: 'raw_counts'
ccc_celltype
CD4_unassessed    190142
Myeloid            93435
Keratinocyte       88692
CD4_malignant      76349
CD4_reactive       72207
Fibroblast         66760
CD8                51831
Vascular           44700
B                  19388
Plasma             10511
Melanocyte         10056
Tregs               1936
Mast                1149
invariants OK: (727156, 1832)
consensus: 4624 interactions over 1877 subunit genes
  1832/1877 genes in data; 4484 interactions fully covered (97.0%)
  control genes missing: none
```

```python  # cell 4
cov = C.coverage_table(adata, sample_key=cd.DONOR_KEY)
display(cov)
cov.to_csv(cd.TAB_DIR / "ccc_coverage_by_donor.csv")
```

```
n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                       
Mast               1149              62            13      0.158537
Tregs              1936              68            20      0.243902
Melanocyte        10056              72            29      0.353659
CD4_malignant     76349              31            29      0.353659
CD4_reactive      72207              36            34      0.414634
B                 19388              70            41      0.500000
Plasma            10511              70            44      0.536585
Keratinocyte      88692              76            62      0.756098
CD8               51831              81            65      0.792683
Fibroblast        66760              81            69      0.841463
Vascular          44700              82            73      0.890244
CD4_unassessed   190142              82            79      0.963415
Myeloid           93435              82            79      0.963415
```

```python  # cell 5
# Balanced subsample is the DEFAULT, for two reasons. Compute: 1000 permutations over ~400k
# cells is slow. Bias: liana's permutation p shrinks with group size, so an unbalanced
# CD4_malignant (76,349 cells vs Tregs' 1,936) scores p=0 on nearly every pair purely on n.
# The per-donor cap additionally stops li2024 (56% of skin cells) from dominating each level.
bal, sub_report = C.subsample_levels(adata)
sub_report.to_csv(cd.TAB_DIR / "ccc_subsample_report.csv", index=False)
print("\ndonors retained per level:")
print(bal.obs.groupby(bal.obs[cd.GROUPBY].astype(str), observed=True)[cd.DONOR_KEY]
      .nunique().sort_values(ascending=False).to_string())
```

```
subsampled 727156 -> 155778 cells
         level  n_before  n_after
CD4_unassessed    190142    15000
       Myeloid     93435    15000
  Keratinocyte     88692    15000
 CD4_malignant     76349    15000
  CD4_reactive     72207    15000
    Fibroblast     66760    15000
           CD8     51831    15000
      Vascular     44700    15000
             B     19388    12917
        Plasma     10511     9973
    Melanocyte     10056     9803
         Tregs      1936     1936
          Mast      1149     1149

donors retained per level:
ccc_celltype
Vascular          82
Myeloid           81
CD4_unassessed    81
CD8               80
Fibroblast        80
Keratinocyte      75
Melanocyte        72
B                 70
Plasma            70
Tregs             68
Mast              62
CD4_reactive      36
CD4_malignant     31
```

## §2 · Core axis — malignant CD4 ↔ CD8 / Myeloid / Fibroblast

The primary run: 24–28 donors clear `MIN_CELLS` on both sides. Both directions are computed
in one call, and the dotplot panels by **sender**.

```python  # cell 7
sub_core, pairs_core, spec_core = C.prepare_analysis(bal, "ctcl_all_core")
res_core = C.run_rank_aggregate(sub_core, resource, groupby_pairs=pairs_core)
res_core.to_csv(cd.TAB_DIR / "ccc_liana_ctcl_all_core.csv", index=False)
print(res_core.shape)
display(res_core.sort_values("magnitude_rank").head(20))
```

```
Using the `lognorm` layer!
20 features of mat are empty, they will be removed.

window: 146806 cells, 73 donors, 90 samples, 7 studies
donors per level:
ccc_celltype
CD4_unassessed    73
Myeloid           73
Vascular          73
Fibroblast        72
CD8               71
Keratinocyte      67
Tregs             66
B                 65
Plasma            64
Melanocyte        63
Mast              56
CD4_reactive      33
CD4_malignant     31

ctcl_all_core: ['CD4_malignant'] <-> ['CD8', 'Myeloid', 'Fibroblast']
  The primary descriptive run: malignant CD4 <-> CD8 / Myeloid / Fibroblast over all CTCL skin, all 7 studies. Pooling studies is defensible here because the axis is malignant-CD4 vs TME *within* the same tissue, not a between-condition comparison. 24-32 donors clear MIN_CELLS on both sides.
Using provided `resource`.

0.03 of entities in the resource are missing from the data.

Generating ligand-receptor stats for 57272 samples and 1808 features
Assuming that counts were `natural` log-normalized!
Running CellPhoneDB

100%|██████████| 1000/1000 [01:31<00:00, 10.93it/s]

Running Connectome
Running log2FC
Running NATMI
Running SingleCellSignalR
(919, 13)

            source         target ligand_complex receptor_complex  lr_means  \
741     Fibroblast  CD4_malignant            VIM             CD44  2.978544   
918        Myeloid  CD4_malignant            VIM             CD44  2.651977   
107  CD4_malignant            CD8            VIM             CD44  2.623429   
282  CD4_malignant     Fibroblast            VIM             CD44  2.622042   
82   CD4_malignant            CD8            MIF       CD74_CXCR4  2.536185   
675     Fibroblast  CD4_malignant         LGALS1            PTPRC  2.508708   
454  CD4_malignant        Myeloid            VIM             CD44  2.492102   
853        Myeloid  CD4_malignant         LGALS1            PTPRC  2.416286   
597     Fibroblast  CD4_malignant         COL6A2             CD44  2.416197   
590     Fibroblast  CD4_malignant         COL1A2             CD44  2.402081   
69   CD4_malignant            CD8         LGALS1            PTPRC  2.338123   
589     Fibroblast  CD4_malignant         COL1A1             CD44  2.336481   
567     Fibroblast  CD4_malignant             C3           IFITM1  2.308175   
81   CD4_malignant            CD8            MIF        CD44_CD74  2.281845   
219  CD4_malignant     Fibroblast            MIF        CD44_CD74  2.258181   
608     Fibroblast  CD4_malignant         CXCL14            CXCR4  2.221647   
596     Fibroblast  CD4_malignant         COL6A1             CD44  2.166295   
403  CD4_malignant        Myeloid            MIF        CD44_CD74  2.150517   
546            CD8  CD4_malignant            VIM             CD44  2.100680   
598     Fibroblast  CD4_malignant         COL6A3             CD44  1.920063   

     cellphone_pvals  expr_prod  scaled_weight  lr_logfc  spec_weight  \
741              0.0   7.173540       0.318479  0.534119     0.410273   
918              0.0   6.079281       0.070900 -0.037955     0.347689   
107              0.0   6.130781       0.054519  0.092920     0.351014   
282              0.0   6.121096       0.053284 -0.088953     0.350460   
82               0.0   6.358780       0.648916  1.581250     0.692341   
675              0.0   6.178295       0.471477  0.615052     0.425618   
454              1.0   5.214017      -0.062087 -0.219761     0.298526   
853              0.0   5.777343       0.407887  0.497267     0.397997   
597              0.0   5.289230       0.854579  2.346273     0.950109   
590              0.0   5.241930       0.824220  2.689984     0.970548   
69               0.0   5.448493       0.363706  0.811840     0.675997   
589              0.0   5.022117       0.763229  2.911715     0.963659   
567              0.0   5.305204       0.942260  2.641739     0.964933   
81               0.0   4.930807       0.284081  0.643995     0.351014   
219              1.0   4.797952      -0.115081 -0.706234     0.169078   

... [1413 more chars truncated]
```

```python  # cell 8
C.dotplot_axis(liana_res=res_core, source_labels=[cd.CD4_MALIGNANT],
               title="malignant CD4 -> core TME")
```

```
<plotnine.ggplot.ggplot object at 0x1521db03bd90>
```

```python  # cell 9
C.dotplot_axis(liana_res=res_core, target_labels=[cd.CD4_MALIGNANT],
               title="core TME -> malignant CD4")
```

```
<plotnine.ggplot.ggplot object at 0x1521867dadd0>
```

## §3 · Extended partners — keratinocyte, vascular, B, plasma, Tregs

```python  # cell 11
sub_ext, pairs_ext, spec_ext = C.prepare_analysis(bal, "ctcl_all_extended")
res_ext = C.run_rank_aggregate(sub_ext, resource, groupby_pairs=pairs_ext, key_added="liana_ext")
res_ext.to_csv(cd.TAB_DIR / "ccc_liana_ctcl_all_extended.csv", index=False)
C.dotplot_axis(liana_res=res_ext, source_labels=[cd.CD4_MALIGNANT],
               title="malignant CD4 -> extended TME")
```

```
Using the `lognorm` layer!
20 features of mat are empty, they will be removed.

window: 146806 cells, 73 donors, 90 samples, 7 studies
donors per level:
ccc_celltype
CD4_unassessed    73
Myeloid           73
Vascular          73
Fibroblast        72
CD8               71
Keratinocyte      67
Tregs             66
B                 65
Plasma            64
Melanocyte        63
Mast              56
CD4_reactive      33
CD4_malignant     31

ctcl_all_extended: ['CD4_malignant'] <-> ['Keratinocyte', 'Vascular', 'B', 'Plasma', 'Tregs']
  Keratinocyte / Vascular / B / Plasma / Tregs. B (21 donors) carries the CXCL13-CXCR5 TLS hypothesis; Tregs (11 donors) is report-only.
Using provided `resource`.

0.03 of entities in the resource are missing from the data.

Generating ligand-receptor stats for 65634 samples and 1808 features
Assuming that counts were `natural` log-normalized!
Running CellPhoneDB

100%|██████████| 1000/1000 [01:05<00:00, 15.35it/s]

Running Connectome
Running log2FC
Running NATMI
Running SingleCellSignalR

<plotnine.ggplot.ggplot object at 0x15218a61f390>
```

```python  # cell 12
C.dotplot_axis(liana_res=res_ext, target_labels=[cd.CD4_MALIGNANT],
               title="extended TME -> malignant CD4")
```

```
<plotnine.ggplot.ggplot object at 0x15218bd3bd10>
```

## §4 · Thin partners — mast, melanocyte

Run so that absence is on the record rather than unexamined. Mast clears `MIN_CELLS` alongside
malignant CD4 in **4 donors** and melanocyte in 13; `axis_feasibility` marks both
`report_only`, and neither appears in nb36.

```python  # cell 14
sub_thin, pairs_thin, _ = C.prepare_analysis(bal, "ctcl_all_thin")
res_thin = C.run_rank_aggregate(sub_thin, resource, groupby_pairs=pairs_thin, key_added="liana_thin")
res_thin.to_csv(cd.TAB_DIR / "ccc_liana_ctcl_all_thin.csv", index=False)
display(res_thin.sort_values("magnitude_rank").head(10))
```

```
Using the `lognorm` layer!

window: 146806 cells, 73 donors, 90 samples, 7 studies
donors per level:
ccc_celltype
CD4_unassessed    73
Myeloid           73
Vascular          73
Fibroblast        72
CD8               71
Keratinocyte      67
Tregs             66
B                 65
Plasma            64
Melanocyte        63
Mast              56
CD4_reactive      33
CD4_malignant     31

ctcl_all_thin: ['CD4_malignant'] <-> ['Mast', 'Melanocyte']
  Mast (5 donors) and Melanocyte (29). Run so that absence is on the record; below MIN_SAMPLES, so nothing here enters a headline figure.
Using provided `resource`.

20 features of mat are empty, they will be removed.
0.03 of entities in the resource are missing from the data.

Generating ligand-receptor stats for 24484 samples and 1808 features
Assuming that counts were `natural` log-normalized!
Running CellPhoneDB

100%|██████████| 1000/1000 [00:28<00:00, 34.50it/s]

Running Connectome
Running log2FC
Running NATMI
Running SingleCellSignalR

            source         target ligand_complex receptor_complex  lr_means  \
335     Melanocyte  CD4_malignant            VIM             CD44  3.260107   
161  CD4_malignant     Melanocyte            VIM             CD44  2.818097   
303     Melanocyte  CD4_malignant         LGALS1            PTPRC  2.634150   
222           Mast  CD4_malignant            VIM             CD44  2.676544   
49   CD4_malignant           Mast            VIM             CD44  2.586301   
31   CD4_malignant           Mast            MIF        CD44_CD74  2.141830   
310     Melanocyte  CD4_malignant            MIF        CD44_CD74  2.063855   
124  CD4_malignant     Melanocyte            PKM             CD44  2.054462   
121  CD4_malignant     Melanocyte            MIF        CD44_CD74  2.001298   
204           Mast  CD4_malignant            MIF        CD44_CD74  1.981866   

     cellphone_pvals  expr_prod  scaled_weight  lr_logfc  spec_weight  \
335              0.0   8.117002       0.281076  0.608213     0.568473   
161              1.0   7.489707      -0.048346 -0.580489     0.560552   
303              0.0   6.722489       0.510645  1.920677     0.655894   
222              1.0   6.161599      -0.197406 -0.495177     0.431527   
49               1.0   5.871599      -0.270921 -0.670404     0.439448   
31               1.0   4.144709      -0.075905  0.370429     0.552598   
310              1.0   4.108602      -0.166804 -0.486312     0.517295   
124              0.0   4.212469       0.244989  0.600172     0.560552   
121              1.0   3.355701      -0.163248  0.262742     0.447402   
204              1.0   3.833871      -0.235518 -0.152289     0.482705   

      lrscore  specificity_rank  magnitude_rank  
335  0.963224          0.467856        0.000080  
161  0.961772          1.000000        0.000318  
303  0.959735          0.051028        0.001265  
222  0.958018          1.000000        0.001265  
49   0.957038          1.000000        0.001973  
31   0.949279          1.000000        0.005021  
310  0.949068          1.000000        0.006342  
124  0.949668          0.467856        0.007814  
121  0.943948          1.000000        0.011206  
204  0.947369          1.000000        0.013125
```

## §5 · Malignant CD4 ↔ reactive CD4

29 donors. Sender and receiver share a lineage, so shared-gene and autocrine artifacts are at
their maximum here: a pair whose ligand is expressed at similar proportion in *both* levels is
a CD4 property, not a directional interaction. §11 prints those proportions.

Both levels are now defined by the same TCR measurement — malignant is "in the ALICE clone
family", reactive is "tested and not in it" — so the axis no longer mixes a CNV-derived call on
one side with its absence on the other.

```python  # cell 16
sub_rea, pairs_rea, spec_rea = C.prepare_analysis(bal, "ctcl_all_reactive")
res_rea = C.run_rank_aggregate(sub_rea, resource, groupby_pairs=pairs_rea, key_added="liana_rea")
res_rea.to_csv(cd.TAB_DIR / "ccc_liana_ctcl_all_reactive.csv", index=False)
C.dotplot_axis(liana_res=res_rea, title="malignant CD4 <-> reactive CD4")
```

```
Using the `lognorm` layer!
20 features of mat are empty, they will be removed.

window: 146806 cells, 73 donors, 90 samples, 7 studies
donors per level:
ccc_celltype
CD4_unassessed    73
Myeloid           73
Vascular          73
Fibroblast        72
CD8               71
Keratinocyte      67
Tregs             66
B                 65
Plasma            64
Melanocyte        63
Mast              56
CD4_reactive      33
CD4_malignant     31

ctcl_all_reactive: ['CD4_malignant'] <-> ['CD4_reactive']
  Malignant CD4 <-> reactive CD4, 29 donors. Sender and receiver share a lineage, so shared-gene and autocrine artifacts are maximal: read only pairs whose ligand is asymmetric between the two levels.
Using provided `resource`.

0.03 of entities in the resource are missing from the data.

Generating ligand-receptor stats for 29566 samples and 1808 features
Assuming that counts were `natural` log-normalized!
Running CellPhoneDB

100%|██████████| 1000/1000 [00:17<00:00, 55.88it/s]

Running Connectome
Running log2FC
Running NATMI
Running SingleCellSignalR

<plotnine.ggplot.ggplot object at 0x1521868acf10>
```

## §6 · The comparator, and the rank-delta table

Same partners, **reactive** CD4 as the sender. No claim of the form *"malignant CD4 signals X to
Y"* is reportable unless the pair is absent or clearly weaker on this axis — otherwise it is a
statement about CD4 T cells in inflamed skin.

Because the atlas has no usable normal-skin T cells (HC skin: 1,524 CD4 across 9 donors), this
is the only available baseline, and it is a *lesional* reactive population. So the claim this
supports is narrower and better defined: **"differs from reactive CD4 in the same lesion."**

```python  # cell 18
sub_cmp, pairs_cmp, spec_cmp = C.prepare_analysis(bal, "ctcl_all_comparator")
res_cmp = C.run_rank_aggregate(sub_cmp, resource, groupby_pairs=pairs_cmp, key_added="liana_cmp")
res_cmp.to_csv(cd.TAB_DIR / "ccc_liana_ctcl_all_comparator.csv", index=False)

sub_cmpx, pairs_cmpx, _ = C.prepare_analysis(bal, "ctcl_all_comparator_extended")
res_cmpx = C.run_rank_aggregate(sub_cmpx, resource, groupby_pairs=pairs_cmpx, key_added="liana_cmpx")
res_cmpx.to_csv(cd.TAB_DIR / "ccc_liana_ctcl_all_comparator_extended.csv", index=False)
print(res_cmp.shape, res_cmpx.shape)
```

```
Using the `lognorm` layer!
20 features of mat are empty, they will be removed.

window: 146806 cells, 73 donors, 90 samples, 7 studies
donors per level:
ccc_celltype
CD4_unassessed    73
Myeloid           73
Vascular          73
Fibroblast        72
CD8               71
Keratinocyte      67
Tregs             66
B                 65
Plasma            64
Melanocyte        63
Mast              56
CD4_reactive      33
CD4_malignant     31

ctcl_all_comparator: ['CD4_reactive'] <-> ['CD8', 'Myeloid', 'Fibroblast']
  THE comparator. No claim of the form 'malignant CD4 signals X to Y' is reportable unless the same pair is absent or weaker with reactive CD4 as the sender in the same tissue. Paired with the rank-delta table, not read on its own.
Using provided `resource`.

0.03 of entities in the resource are missing from the data.

Generating ligand-receptor stats for 56838 samples and 1808 features
Assuming that counts were `natural` log-normalized!
Running CellPhoneDB

100%|██████████| 1000/1000 [00:58<00:00, 16.98it/s]

Running Connectome
Running log2FC
Running NATMI
Running SingleCellSignalR

Using the `lognorm` layer!
20 features of mat are empty, they will be removed.

window: 146806 cells, 73 donors, 90 samples, 7 studies
donors per level:
ccc_celltype
CD4_unassessed    73
Myeloid           73
Vascular          73
Fibroblast        72
CD8               71
Keratinocyte      67
Tregs             66
B                 65
Plasma            64
Melanocyte        63
Mast              56
CD4_reactive      33
CD4_malignant     31

ctcl_all_comparator_extended: ['CD4_reactive'] <-> ['Keratinocyte', 'Vascular', 'B', 'Plasma', 'Tregs']
  Comparator for the extended partners; same reading rule.
Using provided `resource`.

0.03 of entities in the resource are missing from the data.

Generating ligand-receptor stats for 65200 samples and 1808 features
Assuming that counts were `natural` log-normalized!
Running CellPhoneDB

100%|██████████| 1000/1000 [01:02<00:00, 15.88it/s]

Running Connectome
Running log2FC
Running NATMI
Running SingleCellSignalR
(684, 13) (901, 13)
```

```python  # cell 19
mal_out = pd.concat([res_core, res_ext]).query("source == @cd.CD4_MALIGNANT")
rea_out = pd.concat([res_cmp, res_cmpx]).query("source == @cd.CD4_REACTIVE")
delta_out = C.rank_delta(mal_out, rea_out)
delta_out.to_csv(cd.TAB_DIR / "ccc_malignant_vs_reactive_rank_delta_outgoing.csv", index=False)

mal_in = pd.concat([res_core, res_ext]).query("target == @cd.CD4_MALIGNANT").rename(
    columns={"source": "target", "target": "source"})
rea_in = pd.concat([res_cmp, res_cmpx]).query("target == @cd.CD4_REACTIVE").rename(
    columns={"source": "target", "target": "source"})
delta_in = C.rank_delta(mal_in, rea_in)
delta_in.to_csv(cd.TAB_DIR / "ccc_malignant_vs_reactive_rank_delta_incoming.csv", index=False)

print("outgoing -- most malignant-shifted (negative delta = better rank from malignant CD4):")
display(delta_out.head(25))
print("\nfound only with malignant CD4 as sender:", int(delta_out['malignant_only'].sum()))
```

```
outgoing -- most malignant-shifted (negative delta = better rank from malignant CD4):

            target ligand_complex receptor_complex  magnitude_rank_malignant  \
142            CD8         LGALS3             LAG3                  0.213679   
1013      Vascular         LGALS3             MCAM                  0.184628   
990       Vascular          GNAI2              F2R                  0.387660   
238     Fibroblast          GNAI2            EDNRB                  0.401299   
1021      Vascular         LRPAP1             LDLR                  0.408124   
99             CD8          GNAI2             CCR5                  0.424797   
40               B         LRPAP1            SORL1                  0.429296   
22               B          GNAI2            S1PR1                  0.445146   
1078      Vascular        TNFSF10        TNFRSF10D                  0.451188   
490   Keratinocyte          TIMP1             CD63                  0.369848   
623        Myeloid         LRPAP1             LRP1                  0.463143   
427   Keratinocyte           JAML            CXADR                  0.455407   
897          Tregs         LRPAP1            SORL1                  0.492875   
960       Vascular          ARPC5            ADRB2                  0.391328   
188     Fibroblast          ANXA1             EGFR                  0.388050   
491   Keratinocyte          TIMP1            FGFR2                  0.499341   
612        Myeloid         LGALS1            ITGB1                  0.112535   
799         Plasma          TIMP1             CD63                  0.218921   
311     Fibroblast          PTPN6             EGFR                  0.509716   
951       Vascular          ANXA1             DYSF                  0.322139   
140            CD8         LGALS1            ITGB1                  0.098834   
962       Vascular            BSG          SLC16A7                  0.512443   
756         Plasma         LGALS1             CD69                  0.195779   
242     Fibroblast          GNAI2            S1PR3                  0.515748   
952       Vascular          ANXA2            ROBO4                  0.364960   

      lr_means_malignant  magnitude_rank_reactive  lr_means_reactive  \
142             1.141429                 0.927691           0.778500   
1013            1.058843                 0.857317           0.695914   
990             0.483237                 0.996554           0.375202   
238             0.493778                 1.000000           0.385743   
1021            0.457152                 1.000000           0.335029   
99              0.491362                 1.000000           0.383327   
40              0.442812                 1.000000           0.320689   
22              0.476652                 1.000000           0.368618   
1078            0.437142                 1.000000           0.295036   
490             0.879566                 0.908266           0.638109   
623             0.469182                 1.000000           0.347060   
427             0.821114                 0.966228           0.643245   
897             0.469631                 1.000000           0.347509   
960             0.594286                 0.894821           0.458413   
188             0.965795                 0.891075           0.799587   
491             0.522440                 1.000000           0.280983   
612             1.334332                 0.608616           0.944458   
799             1.016402                 0.710604           0.774945   
311             0.460573                 1.000000           0.347459   
951             0.912964                 0.812027           0.746755   
140             1.350811                 0.587097           0.960937   
962             0.481496                 1.000000           0.371053   
756             1.236242                 0.680563           0.846368   
242             0.483319                 1.000000           0.375284   
952             0.881041        
... [1412 more chars truncated]
```

```python  # cell 20
print("incoming -- TME pairs most shifted toward malignant CD4 as the receiver:")
display(delta_in.head(25))
```

```
incoming -- TME pairs most shifted toward malignant CD4 as the receiver:

            target ligand_complex receptor_complex  magnitude_rank_malignant  \
330     Fibroblast           IL15            IL2RA                  0.125520   
672        Myeloid           IL15            IL2RA                  0.445951   
853         Plasma          TIMP1             CD63                  0.397185   
1089      Vascular           IL15            IL2RA                  0.454257   
970          Tregs          TIMP1             CD63                  0.459330   
458   Keratinocyte          CXADR             JAML                  0.455407   
631        Myeloid          HBEGF             CD82                  0.411633   
325     Fibroblast          ICAM1            IL2RA                  0.403866   
741        Myeloid          TOR2A          ATP5F1B                  0.468803   
881          Tregs           CD47            SIRPG                  0.529140   
506   Keratinocyte          ICAM1            IL2RA                  0.542980   
105              B          TGFB1           TGFBR3                  0.471725   
724        Myeloid          TIMP1             CD63                  0.197889   
132            CD8           CD55           ADGRE5                  0.581982   
379     Fibroblast            PTN              NCL                  0.341367   
88               B         LGALS9           SLC1A5                  0.591034   
482   Keratinocyte          HLA-A            APLP2                  0.422687   
437   Keratinocyte           BMP2            ACTR2                  0.548497   
511   Keratinocyte           JAG1             CD46                  0.600776   
769         Plasma           CD47            SIRPG                  0.565963   
100              B         PTDSS1            JMJD6                  0.608153   
194            CD8         SELPLG             SELL                  0.575443   
86               B         LGALS9             CD47                  0.560101   
1054      Vascular          HLA-A            APLP2                  0.384665   
968          Tregs          TGFB1           TGFBR3                  0.618082   

      lr_means_malignant  magnitude_rank_reactive  lr_means_reactive  \
330             0.160535                 0.819389           0.279368   
672             0.193268                 1.000000           0.312101   
853             0.857959                 0.944775           0.622885   
1089            0.179274                 1.000000           0.298108   
970             0.566291                 1.000000           0.331217   
458             0.821114                 0.966228           0.643245   
631             0.542588                 0.918450           0.421194   
325             0.604291                 0.900636           0.723124   
741             0.720282                 0.954781           0.587001   
881             0.433139                 1.000000           0.361028   
506             0.184911                 1.000000           0.303744   
105             0.432850                 0.926400           0.381929   
724             1.154207                 0.652426           0.919133   
132             0.437479                 1.000000           0.386863   
379             0.988641                 0.754490           0.871459   
88              0.396511                 1.000000           0.301713   
482             0.919702                 0.825353           0.901626   
437             0.459619                 0.949406           0.425285   
511             0.391800                 1.000000           0.362807   
769             0.441551                 0.961050           0.369440   
100             0.388419                 1.000000           0.299023   
194             0.500008                 0.967100           0.677623   
86              0.400862                 0.949406           0.370483   
1054            0.955409                 0.771865           0.937333   
968             0.387596                 1.00
... [1351 more chars truncated]
```

## §7 · Controls

Positives are drawn from `docs/CTCL_atlas_notebook_replication_spec.md` (Fig 5f/6f/8a) and the
CTCL literature; the three the spec names explicitly (`CXCL13–CXCR5`, `CD40LG–CD40`,
`CD86–CD28` on the malignant↔B axis) are held to a stricter bar. Negatives are
lineage-impossible edges — a T cell does not send `KITLG` or `COL1A1`.

`MIF–CD74` sitting at the top is the **documented failure mode**, not a finding: MIF is
ubiquitous and very highly expressed, so it is the positive control for the null model breaking.

```python  # cell 22
res_all = pd.concat([res_core, res_ext, res_rea], ignore_index=True)
ctrl, caveat_hits = C.control_report(res_all, resource=resource, top_n=40)
ctrl.to_csv(cd.TAB_DIR / "ccc_control_report.csv", index=False)
caveat_hits.to_csv(cd.TAB_DIR / "ccc_caveat_hits.csv", index=False)
display(ctrl)
```

```
positives found 10/12 (0 in top 40); negatives found 0/3 (want 0 or bottom-decile)

caveat genes in the top 10 -- documented failure mode, not biology:
gene       interaction        source target
 MIF MIF -> CD74_CXCR4 CD4_malignant      B
CD74 MIF -> CD74_CXCR4 CD4_malignant      B

        kind         source         target ligand_complex  \
0   positive  CD4_malignant              B         CXCL13   
1   positive  CD4_malignant              B         CD40LG   
2   positive              B  CD4_malignant           CD86   
3   positive              B  CD4_malignant            TNF   
4   positive        Myeloid  CD4_malignant           CD86   
5   positive        Myeloid  CD4_malignant          CCL17   
6   positive        Myeloid  CD4_malignant          CCL22   
7   positive        Myeloid  CD4_malignant           CD58   
8   positive     Fibroblast  CD4_malignant         CXCL12   
9   positive  CD4_malignant        Myeloid      CSF1_IL34   
10  positive  CD4_malignant     Fibroblast          TGFB1   
11  positive  CD4_malignant     Fibroblast          TGFB1   
12  negative  CD4_malignant     Fibroblast          KITLG   
13  negative  CD4_malignant   Keratinocyte         COL1A1   
14  negative  CD4_malignant   Keratinocyte           KRT1   

       receptor_complex  found  rank_position  magnitude_rank  \
0                 CXCR5   True         2172.0        1.000000   
1                  CD40   True         1459.0        0.762877   
2                  CD28  False            NaN             NaN   
3              TNFRSF1B   True          645.0        0.433223   
4                  CD28   True         1220.0        0.658991   
5                  CCR4   True         1037.0        0.585270   
6                  CCR4   True          850.0        0.515748   
7                   CD2   True         1062.0        0.596466   
8                 CXCR4   True          177.0        0.061165   
9                 CSF1R  False            NaN             NaN   
10        TGFBR1_TGFBR2   True          532.0        0.378684   
11  ACVR1_TGFBR1_TGFBR2   True         1502.0        0.779792   
12                  KIT  False            NaN             NaN   
13          ITGA2_ITGB1  False            NaN             NaN   
14                    *  False            NaN             NaN   

    specificity_rank  lr_means  cellphone_pvals in_top_n  pct_of_tested  
0           0.001225  0.553925              0.0    False       0.923077  
1           0.082745  0.598214              0.0    False       0.620059  
2                NaN       NaN              NaN      NaN            NaN  
3           0.099788  0.442563              0.0    False       0.274118  
4           0.024647  0.516671              0.0    False       0.518487  
5           0.062325  0.434937              0.0    False       0.440714  
6           0.030609  0.477805              0.0    False       0.361241  
7           0.126063  0.851854              0.0    False       0.451339  
8           0.022464  1.512675              0.0    False       0.075223  
9                NaN       NaN              NaN      NaN            NaN  
10          0.374281  0.536301              0.0    False       0.226094  
11          0.111490  0.503005              0.0    False       0.638334  
12               NaN       NaN              NaN      NaN            NaN  
13               NaN       NaN              NaN      NaN            NaN  
14               NaN       NaN              NaN      NaN            NaN
```

```python  # cell 23
spec_ctrl = C.resolve_controls(cd.SPEC_MUST_HAVES, resource)
found = ctrl.merge(pd.DataFrame(spec_ctrl, columns=["source", "target", "ligand_complex", "receptor_complex"]),
                   on=["source", "target", "ligand_complex", "receptor_complex"])
print("replication-spec must-haves:")
display(found)
if len(found) and "pct_of_tested" in found:
    print("in the top quartile of magnitude_rank:", (found['pct_of_tested'] < 0.25).sum(), "/", len(found))
```

```
replication-spec must-haves:

       kind         source         target ligand_complex receptor_complex  \
0  positive  CD4_malignant              B         CXCL13            CXCR5   
1  positive  CD4_malignant              B         CD40LG             CD40   
2  positive              B  CD4_malignant           CD86             CD28   

   found  rank_position  magnitude_rank  specificity_rank  lr_means  \
0   True         2172.0        1.000000          0.001225  0.553925   
1   True         1459.0        0.762877          0.082745  0.598214   
2  False            NaN             NaN               NaN       NaN   

   cellphone_pvals in_top_n  pct_of_tested  
0              0.0    False       0.923077  
1              0.0    False       0.620059  
2              NaN      NaN            NaN  
in the top quartile of magnitude_rank: 0 / 3
```

## §8 · `expr_prop` sensitivity

`expr_prop` is the single biggest lever on the result, and the low-abundance cytokines
(`IL13`, `IL4`, `IL31`, `IL2`) sit right at the threshold. Anything not stable across
0.05 / 0.1 / 0.2 is a threshold artifact.

```python  # cell 25
sweep, stable = C.expr_prop_sweep(sub_core, resource, groupby_pairs=pairs_core, verbose=False)
sweep.to_csv(cd.TAB_DIR / "ccc_expr_prop_sweep_core.csv", index=False)
print(f"{len(stable)}/{cd.TOP_N} of the top-{cd.TOP_N} are stable across expr_prop "
      f"{cd.EXPR_PROP_SWEEP}")
display(pd.DataFrame(sorted(stable), columns=["source", "target", "ligand_complex", "receptor_complex"]))
```

```
19/20 of the top-20 are stable across expr_prop [0.05, 0.1, 0.2]

           source         target ligand_complex receptor_complex
0   CD4_malignant            CD8         LGALS1            PTPRC
1   CD4_malignant            CD8            MIF        CD44_CD74
2   CD4_malignant            CD8            MIF       CD74_CXCR4
3   CD4_malignant            CD8            VIM             CD44
4   CD4_malignant     Fibroblast            MIF        CD44_CD74
5   CD4_malignant     Fibroblast            VIM             CD44
6   CD4_malignant        Myeloid            MIF        CD44_CD74
7   CD4_malignant        Myeloid            VIM             CD44
8             CD8  CD4_malignant            VIM             CD44
9      Fibroblast  CD4_malignant             C3           IFITM1
10     Fibroblast  CD4_malignant         COL1A1             CD44
11     Fibroblast  CD4_malignant         COL1A2             CD44
12     Fibroblast  CD4_malignant         COL6A1             CD44
13     Fibroblast  CD4_malignant         COL6A2             CD44
14     Fibroblast  CD4_malignant         CXCL14            CXCR4
15     Fibroblast  CD4_malignant         LGALS1            PTPRC
16     Fibroblast  CD4_malignant            VIM             CD44
17        Myeloid  CD4_malignant         LGALS1            PTPRC
18        Myeloid  CD4_malignant            VIM             CD44
```

## §9 · Within-donor label shuffle

The grouping label is permuted **within each donor**, so every donor's cell-type composition is
held fixed and what dissolves is cell-type specificity rather than donor identity. A global
shuffle would also destroy composition and pass trivially.

Target: **≤ 2/20** overlap with the real top-20.

```python  # cell 27
shuf, overlap = C.shuffle_control(sub_core, resource, groupby_pairs=pairs_core,
                                  reference=res_core, verbose=False)
overlap.to_csv(cd.TAB_DIR / "ccc_shuffle_control.csv", index=False)
print("\nmean overlap:", overlap['overlap_with_real'].mean(),
      "-- target <= 2 of", cd.TOP_N)
```

```
shuffle  overlap_with_real  top_n
       0                 13     20
       1                 13     20
       2                 13     20

mean overlap: 13.0 -- target <= 2 of 20
```

## §10 · Downsample stability

Makes the n-dependence of the permutation p-value explicit instead of leaving it as an unstated
property of the default subsample. Watch `frac_pval_zero` grow with `n_cells`: that is the
pseudoreplication, measured.

```python  # cell 29
stab, tops = C.downsample_stability(sub_core, resource, groupby_pairs=pairs_core)
stab.to_csv(cd.TAB_DIR / "ccc_downsample_stability.csv", index=False)
display(stab)
```

```
max_per_level  n_cells  median_cellphone_pval  frac_pval_zero  overlap_with_largest
          5000    57922                    0.0        0.670996                    20
         15000   146806                    0.0        0.694233                    20
         40000   146806                    0.0        0.694233                    20

   max_per_level  n_cells  median_cellphone_pval  frac_pval_zero  \
0           5000    57922                    0.0        0.670996   
1          15000   146806                    0.0        0.694233   
2          40000   146806                    0.0        0.694233   

   overlap_with_largest  
0                    20  
1                    20  
2                    20
```

## §11 · Forced curated panel

Every curated pair is reported whether or not it cleared `expr_prop`, with the actual detection
proportion in sender and receiver. A pair that failed is then reported as *"ligand detected in
6 % of the sender"* rather than as absent — a coverage limit must never be read as biology.

This is also where the malignant↔reactive autocrine caveat from §5 is checkable: compare the
ligand's `expr_prop` in `CD4_malignant` against `CD4_reactive`.

```python  # cell 31
for gene, note in cd.LR_CAVEATS.items():
    print(f"{gene}: {note}\n")
```

```
MIF: MIF is ubiquitous and very highly expressed; a top-ranked MIF->CD74 or MIF->CXCR4 edge is a null-model failure until shown otherwise. Treat it as the positive control for the FAILURE mode, not as biology.

B2M: Housekeeping-level expression; any B2M edge is an artifact of the null.

HLA-DRA: MHC-II 'interactions' in the consensus resource largely track myeloid identity rather than a signalling event.

CD74: See MIF and HLA-DRA -- CD74 is the receiver side of both failure modes.

IL13: Th2 cytokine mRNA is poorly captured by 3' 10x, so a negative IL13->IL4R edge is uninformative. That is exactly why it is in the forced panel.

IL4: See IL13.

TNFSF8: CD30L->CD30 marks large-cell transformation; expect it only in advanced or LCT donors, and read its absence as coverage, not biology.

PVR: The consensus resource stores the nectin axis with PVR on the RECEPTOR side (TIGIT -> PVR, CD96 -> PVR), which is backwards biologically: TIGIT and CD96 are the receptors. ccc_helpers.resolve_pairs flips the direction to match the resource, so read these edges as 'the PVR-expressing cell is liana's target' and interpret the biology in the opposite direction.

TIGIT: See PVR -- direction is stored inverted in the resource.
```

```python  # cell 32
panel_res = pd.concat(
    [C.filter_to_panel(r, resource=resource).assign(analysis=name)
     for name, r in [("core", res_core), ("extended", res_ext), ("reactive", res_rea),
                     ("comparator", res_cmp), ("comparator_extended", res_cmpx)]],
    ignore_index=True,
)
panel_res.to_csv(cd.TAB_DIR / "ccc_liana_curated_panel.csv", index=False)
print(f"{len(panel_res)} scored rows over {panel_res.group.nunique()} panel groups")
display(panel_res.groupby("group").size().rename("n_scored").to_frame())
```

```
201 scored rows over 8 panel groups

                         n_scored
group                            
adhesion_migration             16
b_cell_tls_axis                53
checkpoint_restraint           20
chemokine_recruitment          24
costim_tme_to_malignant        58
malignant_to_myeloid           14
survival_signal3                2
th2_stroma_remodelling         14
```

```python  # cell 33
fpe = C.forced_panel_expression(bal, resource=resource)
fpe.to_csv(cd.TAB_DIR / "ccc_forced_panel_expression.csv", index=False)

# which curated genes sit just under the expr_prop threshold in the senders -- these are the
# pairs whose absence from the result is a detection limit, not a negative
focus = fpe[fpe.level.isin([cd.CD4_MALIGNANT, cd.CD4_REACTIVE, "Myeloid", "Fibroblast", "B"])]
near = focus[(focus.expr_prop > 0.01) & (focus.expr_prop < cd.EXPR_PROP)]
print(f"curated genes between 1% and expr_prop={cd.EXPR_PROP} in a focal level:")
display(near.sort_values("expr_prop", ascending=False).head(30))
```

```
curated genes between 1% and expr_prop=0.1 in a focal level:

              level      gene  n_cells  expr_prop  mean_lognorm  \
68                B     ITGAL    12917   0.099946      0.179989   
190   CD4_malignant    TGFBR2    15000   0.098867      0.136883   
939         Myeloid     CCL19    15000   0.098467      0.172247   
994         Myeloid     IL2RA    15000   0.098267      0.146349   
52                B   IL13RA1    12917   0.095998      0.188985   
258    CD4_reactive    IL10RB    15000   0.095733      0.167774   
586      Fibroblast       IL7    15000   0.094800      0.106261   
207   CD4_malignant    TNFSF9    15000   0.093333      0.140567   
151   CD4_malignant      IFNG    15000   0.092467      0.158827   
580      Fibroblast     IL2RG    15000   0.091600      0.127099   
1004        Myeloid     ITGAL    15000   0.091467      0.118069   
286    CD4_reactive     PDCD1    15000   0.089800      0.156069   
153   CD4_malignant    IFNGR2    15000   0.089333      0.121296   
281    CD4_reactive       LTA    15000   0.087400      0.159768   
23                B      CD86    12917   0.084695      0.155697   
148   CD4_malignant     ICAM1    15000   0.082867      0.111719   
112   CD4_malignant     CCR10    15000   0.081533      0.112237   
24                B      CD96    12917   0.081288      0.142382   
594      Fibroblast       LTB    15000   0.080933      0.122686   
990         Myeloid    IL15RA    15000   0.080000      0.082521   
551      Fibroblast    CXCL10    15000   0.079667      0.184039   
599      Fibroblast  PDCD1LG2    15000   0.079000      0.084658   
561      Fibroblast     ERBB2    15000   0.078733      0.081876   
587      Fibroblast      IL7R    15000   0.078533      0.112570   
280    CD4_reactive    LGALS9    15000   0.077800      0.138291   
103               B    TNFSF9    12917   0.077727      0.154894   
946         Myeloid      CCR5    15000   0.077267      0.086674   
12                B       CD2    12917   0.076798      0.138946   
949         Myeloid     CD226    15000   0.072667      0.081507   
307    CD4_reactive   TNFSF14    15000   0.069933      0.122118   

      passes_min_cells  
68                True  
190               True  
939               True  
994               True  
52                True  
258               True  
586               True  
207               True  
151               True  
580               True  
1004              True  
286               True  
153               True  
281               True  
23                True  
148               True  
112               True  
24                True  
594               True  
990               True  
551               True  
599               True  
561               True  
587               True  
280               True  
103               True  
946               True  
12                True  
949               True  
307               True
```

```python  # cell 34
# autocrine check for the malignant <-> reactive axis: a ligand at similar proportion in both
# CD4 levels is a CD4 property, not a directional signal
piv = (focus[focus.level.isin([cd.CD4_MALIGNANT, cd.CD4_REACTIVE])]
       .pivot(index="gene", columns="level", values="expr_prop"))
piv["ratio_mal_over_rea"] = piv[cd.CD4_MALIGNANT] / piv[cd.CD4_REACTIVE].replace(0, np.nan)
print("curated genes most malignant-skewed within CD4:")
display(piv.sort_values("ratio_mal_over_rea", ascending=False).head(20))
print("curated genes symmetric between the two CD4 levels (autocrine artifact risk):")
display(piv[(piv.ratio_mal_over_rea.between(0.8, 1.25)) & (piv[cd.CD4_MALIGNANT] > cd.EXPR_PROP)]
        .head(20))
```

```
curated genes most malignant-skewed within CD4:

level    CD4_malignant  CD4_reactive  ratio_mal_over_rea
gene                                                    
KLRC1         0.100333      0.008267           12.137097
IL4           0.017867      0.001600           11.166667
IL22          0.118133      0.012467            9.475936
TNFSF4        0.053733      0.010000            5.373333
PVR           0.007267      0.001533            4.739130
ICOSLG        0.005800      0.001267            4.578947
CXCL13        0.218533      0.048200            4.533887
IL13          0.101400      0.023267            4.358166
IL31          0.000533      0.000133            4.000000
IL22RA1       0.001533      0.000400            3.833333
CD70          0.245067      0.067000            3.657711
IL15          0.015733      0.004333            3.630769
CSF2          0.120133      0.035333            3.400000
ICAM1         0.082867      0.025733            3.220207
TNFSF14       0.211467      0.069933            3.023832
CSF2RB        0.022000      0.008000            2.750000
CD86          0.011267      0.004133            2.725806
HAVCR2        0.053733      0.020133            2.668874
SELE          0.002867      0.001133            2.529412
IL13RA1       0.006267      0.002533            2.473684
curated genes symmetric between the two CD4 levels (autocrine artifact risk):

level     CD4_malignant  CD4_reactive  ratio_mal_over_rea
gene                                                     
CCR4           0.212933      0.231133            0.921258
CD2            0.565800      0.567267            0.997415
CD27           0.366267      0.355000            1.031737
CD28           0.259933      0.257533            1.009319
CD40LG         0.167667      0.136067            1.232239
CD44           0.784600      0.729067            1.076170
CTLA4          0.247800      0.280533            0.883317
CXCR4          0.586067      0.667400            0.878134
CXCR6          0.142933      0.156667            0.912340
HLA-E          0.412400      0.417067            0.988811
IL2RG          0.839667      0.693733            1.210359
IL7R           0.599533      0.709800            0.844651
ITGB2          0.419133      0.352133            1.190269
LTB            0.613200      0.625600            0.980179
SELPLG         0.390267      0.341267            1.143583
TIGIT          0.301867      0.297133            1.015930
TNF            0.140333      0.132667            1.057789
TNFRSF14       0.273867      0.249133            1.099277
TNFRSF1B       0.393000      0.345467            1.137592
TNFRSF4        0.373133      0.342067            1.090821
```

## §12 · Study diagnostic

`expr_prop` is a detection-rate threshold and chemistry tracks study (5′ / 3′ / FFPE Flex), so
a pair that passes in one study and fails in another is a capture artifact rather than a
biological difference. Only studies with enough donors on both sides of the core axis are run.

```python  # cell 36
ok_studies = [
    s for s in bal.obs[cd.STUDY_KEY].astype(str).unique()
    if (bal.obs[bal.obs[cd.STUDY_KEY].astype(str) == s]
        .groupby(cd.GROUPBY, observed=True)[cd.DONOR_KEY].nunique()
        .reindex([cd.CD4_MALIGNANT, "Myeloid"]).fillna(0) >= 3).all()
]
print("studies with >= 3 donors on both sides:", ok_studies)
by_study = C.run_by_group(sub_core, resource, cd.STUDY_KEY, groups=ok_studies,
                          groupby_pairs=pairs_core, verbose=False)
by_study.to_csv(cd.TAB_DIR / "ccc_liana_by_study_core.csv", index=False)
```

```
studies with >= 3 donors on both sides: ['chennareddy2025', 'brunner2024', 'herrera2021', 'li2024']
```

```python  # cell 37
if len(by_study):
    named = {s: by_study[by_study[cd.STUDY_KEY] == s] for s in ok_studies}
    ov = C.top_n_overlap_matrix(named)
    print(f"top-{cd.TOP_N} overlap between studies (core axis):")
    display(ov)
    ov.to_csv(cd.TAB_DIR / "ccc_study_topn_overlap.csv")
    keys = ["source", "target", "ligand_complex", "receptor_complex"]
    per = by_study.groupby(keys)[cd.STUDY_KEY].nunique().rename("n_studies")
    top = res_core.sort_values("magnitude_rank").head(30).set_index(keys)
    print("\nhow many studies recover each of the top-30 pooled pairs:")
    display(top.join(per).reset_index()[keys + ["magnitude_rank", "n_studies"]])
```

```
top-20 overlap between studies (core axis):

                 chennareddy2025  brunner2024  herrera2021  li2024
chennareddy2025             20.0         16.0         10.0    11.0
brunner2024                 16.0         20.0          9.0    11.0
herrera2021                 10.0          9.0         20.0     5.0
li2024                      11.0         11.0          5.0    20.0

how many studies recover each of the top-30 pooled pairs:

           source         target ligand_complex receptor_complex  \
0      Fibroblast  CD4_malignant            VIM             CD44   
1         Myeloid  CD4_malignant            VIM             CD44   
2   CD4_malignant            CD8            VIM             CD44   
3   CD4_malignant     Fibroblast            VIM             CD44   
4   CD4_malignant            CD8            MIF       CD74_CXCR4   
5      Fibroblast  CD4_malignant         LGALS1            PTPRC   
6   CD4_malignant        Myeloid            VIM             CD44   
7         Myeloid  CD4_malignant         LGALS1            PTPRC   
8      Fibroblast  CD4_malignant         COL6A2             CD44   
9      Fibroblast  CD4_malignant         COL1A2             CD44   
10  CD4_malignant            CD8         LGALS1            PTPRC   
11     Fibroblast  CD4_malignant         COL1A1             CD44   
12     Fibroblast  CD4_malignant             C3           IFITM1   
13  CD4_malignant            CD8            MIF        CD44_CD74   
14  CD4_malignant     Fibroblast            MIF        CD44_CD74   
15     Fibroblast  CD4_malignant         CXCL14            CXCR4   
16     Fibroblast  CD4_malignant         COL6A1             CD44   
17  CD4_malignant        Myeloid            MIF        CD44_CD74   
18            CD8  CD4_malignant            VIM             CD44   
19     Fibroblast  CD4_malignant         COL6A3             CD44   
20            CD8  CD4_malignant            MIF        CD44_CD74   
21     Fibroblast  CD4_malignant           FADD            TRAF2   
22  CD4_malignant            CD8            PKM             CD44   
23  CD4_malignant     Fibroblast            PKM             CD44   
24        Myeloid  CD4_malignant          HLA-B             CD3D   
25     Fibroblast  CD4_malignant         LGALS1             CD69   
26     Fibroblast  CD4_malignant            MIF        CD44_CD74   
27     Fibroblast  CD4_malignant          TIMP1             CD63   
28  CD4_malignant            CD8         LGALS1             CD69   
29  CD4_malignant        Myeloid            B2M             CD1C   

    magnitude_rank  n_studies  
0         0.000011          3  
1         0.000382          4  
2         0.000678          4  
3         0.001279          3  
4         0.001521          4  
5         0.001784          3  
6         0.002067          4  
7         0.002372          4  
8         0.002696          3  
9         0.003042          3  
10        0.003408          4  
11        0.003794          3  
12        0.004628          3  
13        0.005075          4  
14        0.005543          3  
15        0.006539          3  
16        0.007068          3  
17        0.007616          4  
18        0.008185          4  
19        0.008774          3  
20        0.009382          4  
21        0.009783          1  
22        0.011327          4  
23        0.012015          3  
24        0.012723          3  
25        0.014197          3  
26        0.014964          3  
27        0.014964          3  
28        0.017381          4  
29        0.017381          4
```

## §13 · Malignancy-definition sensitivity

The primary call is **ALICE-TCR alone**, chosen because a TCR-sequence fact is independent of
the expression LIANA scores. The alternatives are not: `mal_cnv` is inferCNV's per-donor cluster
call, derived from *smoothed regional expression*, so a `cnv_only` label and that cell's
expression are partly the same measurement — a pair whose ligand or receptor sits on a
recurrently gained arm could be reporting dosage. `mal_combined` is `mal_cnv ∪ mal_tcr_alice`
and inherits the problem for the 77,624 CD4 it adds on top of ALICE.

This section runs the core axis under all four definitions anyway. The point is no longer to
pick one — it is to show the headline pairs do not depend on the choice. Two things to read:

- **ALICE vs mal_cnv** is the load-bearing comparison. They disagree on 40,447 CD4 that ALICE
  calls reactive and inferCNV calls malignant; if the top pairs survive both, the circularity
  never mattered.
- Any headline pair not stable across **≥3 of the 4 definitions** is demoted.

The CDR3 gate that separates reactive from unassessed is applied to the ALICE primary **only**
(`build_ccc_celltype_from` passes `tcr_assessed_src=None`): "could ALICE test this cell" is
meaningless for a CNV- or paper-label-derived call, and gating those on CDR3 recovery would mix
unrelated missingness mechanisms and make the overlap matrix uninterpretable.

```python  # cell 39
alt_res, alt_counts = {cd.MALIG_SRC: res_core}, {}
for defn in ["mal_combined", "mal_cnv", "tumor_cell"]:
    a = bal.copy()
    a.obs[cd.GROUPBY] = C.build_ccc_celltype_from(a.obs, defn, verbose=False)
    a = a[a.obs[cd.GROUPBY].notna()].copy()
    a.layers[cd.LAYER] = a.X
    alt_counts[defn] = a.obs[cd.GROUPBY].value_counts().to_dict()
    try:
        alt_res[defn] = C.run_rank_aggregate(a, resource, groupby_pairs=pairs_core,
                                             key_added=f"liana_{defn}", verbose=False)
    except Exception as exc:
        print(f"{defn}: {exc}")
print(f"primary = {cd.MALIG_SRC}; alternatives relabelled without the CDR3 gate\n")
print(pd.DataFrame(alt_counts).fillna(0).astype(int).to_string())
```

```
primary = mal_tcr_alice; alternatives relabelled without the CDR3 gate

                mal_combined  mal_cnv  tumor_cell
CD4_malignant          23349    19997       12839
CD8                    15000    15000       15000
Fibroblast             15000    15000       15000
Vascular               15000    15000       15000
Myeloid                15000    15000       15000
Keratinocyte           15000    15000       15000
B                      12917    12917       12917
CD4_reactive           11963    15315       11381
Plasma                  9973     9973        9973
Melanocyte              9803     9803        9803
CD4_unassessed          9688     9688       20780
Tregs                   1936     1936        1936
Mast                    1149     1149        1149
```

```python  # cell 40
ov_def = C.top_n_overlap_matrix(alt_res)
print(f"top-{cd.TOP_N} overlap across malignancy definitions (core axis):")
display(ov_def)
ov_def.to_csv(cd.TAB_DIR / "ccc_malignancy_definition_topn_overlap.csv")

keys = ["source", "target", "ligand_complex", "receptor_complex"]
n_def = (pd.concat([r.sort_values("magnitude_rank").head(cd.TOP_N).assign(defn=k)
                    for k, r in alt_res.items()])
         .groupby(keys)["defn"].nunique().rename("n_definitions"))
n_def.sort_values(ascending=False).to_csv(cd.TAB_DIR / "ccc_pair_definition_stability.csv")
print("\npairs in the top-20 of all four definitions:")
display(n_def[n_def == len(alt_res)].to_frame())
```

```
top-20 overlap across malignancy definitions (core axis):

               mal_tcr_alice  mal_combined  mal_cnv  tumor_cell
mal_tcr_alice           20.0          19.0     19.0        19.0
mal_combined            19.0          20.0     20.0        20.0
mal_cnv                 19.0          20.0     20.0        20.0
tumor_cell              19.0          20.0     20.0        20.0

pairs in the top-20 of all four definitions:

                                                             n_definitions
source        target        ligand_complex receptor_complex               
CD4_malignant CD8           LGALS1         PTPRC                         4
                            MIF            CD44_CD74                     4
                                           CD74_CXCR4                    4
                            VIM            CD44                          4
              Fibroblast    MIF            CD44_CD74                     4
                            VIM            CD44                          4
              Myeloid       MIF            CD44_CD74                     4
                            VIM            CD44                          4
CD8           CD4_malignant VIM            CD44                          4
Fibroblast    CD4_malignant C3             IFITM1                        4
                            COL1A1         CD44                          4
                            COL1A2         CD44                          4
                            COL6A1         CD44                          4
                            COL6A2         CD44                          4
                            CXCL14         CXCR4                         4
                            LGALS1         PTPRC                         4
                            VIM            CD44                          4
Myeloid       CD4_malignant LGALS1         PTPRC                         4
                            VIM            CD44                          4
```

```python  # cell 41
# the evidence-strength view: restrict malignant CD4 to the 62,096 cells that ALSO carry a CNV
# call. Under an ALICE primary this is a NESTED subset of CD4_malignant (62,096 of 76,349), not
# an orthogonal definition -- it asks whether the 14,253 tcr_only cells drive anything.
both = bal[(bal.obs[cd.GROUPBY].astype(str) != cd.CD4_MALIGNANT)
           | (bal.obs[cd.EVIDENCE_SRC].astype(str) == "both")].copy()
both.layers[cd.LAYER] = both.X

is_mal = bal.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT
ev_both = bal.obs[cd.EVIDENCE_SRC].astype(str) == "both"
assert not (ev_both & ~is_mal & (bal.obs[cd.CELLTYPE_SRC].astype(str) == "CD4")).any(), \
    "evidence=='both' CD4 outside CD4_malignant -- 'both' should be nested inside ALICE"
print(f"malignant CD4 in the subsample: {int(is_mal.sum())} "
      f"-> restricted to evidence=='both': {int((is_mal & ev_both).sum())}")
print("donors:", both.obs.loc[both.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT,
                              cd.DONOR_KEY].nunique())
res_both = C.run_rank_aggregate(both, resource, groupby_pairs=pairs_core,
                                key_added="liana_both", verbose=False)
res_both.to_csv(cd.TAB_DIR / "ccc_liana_core_evidence_both.csv", index=False)
display(C.top_n_overlap_matrix({cd.MALIG_SRC: res_core, "evidence_both": res_both}))
```

```
malignant CD4 in the subsample: 15000 -> restricted to evidence=='both': 11648
donors: 31

               mal_tcr_alice  evidence_both
mal_tcr_alice           20.0           19.0
evidence_both           19.0           20.0
```

## §14 · CD8 sensitivity — drop the cnv-only malignant calls

ALICE is a CD4-only caller, so under the primary definition **no CD8 cell is malignant** and the
CD8 level is a single pooled level by construction rather than by choice. That does not make the
question go away: `mal_cnv` flags 21,200 of 51,831 skin CD8 as malignant, all `cnv_only` with
zero TCR support — a genuine CD8 malignancy, a CNV false positive, or ambient bleed from the
dominant CD4 clone. If those cells are transcriptionally odd, the CD8 comparator level is
contaminated whichever call is primary.

So the drop is driven off `mal_combined` explicitly (`build_ccc_celltype` with the primary
would drop nothing), and this run quantifies what the contamination costs.

```python  # cell 43
cd8_alt = bal.copy()
# CD4 levels stay on the ALICE primary; only the CD8 drop reads mal_combined, because that is
# where the cnv_only CD8 flag lives. Build the primary label, then null out those CD8 cells.
cd8_flagged = ((cd8_alt.obs[cd.CELLTYPE_SRC].astype(str) == "CD8")
               & cd8_alt.obs["mal_combined"].fillna(False).astype(bool))
lab = C.build_ccc_celltype(cd8_alt.obs, verbose=False)
cd8_alt.obs[cd.GROUPBY] = lab.mask(cd8_flagged)      # mask -> NaN, categories preserved
print(f"dropped {int(cd8_flagged.sum())} CD8 cells with a cnv_only malignant call")
cd8_alt = cd8_alt[cd8_alt.obs[cd.GROUPBY].notna()].copy()
cd8_alt.obs[cd.GROUPBY] = cd8_alt.obs[cd.GROUPBY].cat.remove_unused_categories()
cd8_alt.layers[cd.LAYER] = cd8_alt.X
res_cd8 = C.run_rank_aggregate(
    cd8_alt, resource,
    groupby_pairs=C.build_groupby_pairs({"ax": ([cd.CD4_MALIGNANT], ["CD8"])}),
    key_added="liana_cd8_clean", verbose=False)
res_cd8.to_csv(cd.TAB_DIR / "ccc_liana_cd8_drop_cnvonly.csv", index=False)

cd8_ref = res_core[res_core[["source", "target"]].isin([cd.CD4_MALIGNANT, "CD8"]).all(axis=1)]
display(C.top_n_overlap_matrix({"cd8_as_is": cd8_ref, "cd8_cnvonly_dropped": res_cd8}))
```

```
dropped 3244 CD8 cells with a cnv_only malignant call

                     cd8_as_is  cd8_cnvonly_dropped
cd8_as_is                 20.0                 17.0
cd8_cnvonly_dropped       17.0                 20.0
```

## §15 · Healthy-skin negative control

HC skin has 27,534 cells but only 1,524 CD4 and 300 CD8 across 9 donors, so **no T-cell axis is
computable** — `CD4_malignant` must be 0 cells here, and that is the expected result rather than
a failure. What is computable is the structural fibroblast → keratinocyte / vascular / myeloid
grid, which should recover plausible matrix and growth-factor edges. If it does not, the problem
is the object, not the biology.

```python  # cell 45
hc, pairs_hc, spec_hc = C.prepare_analysis(adata, "hc_structural")
n_mal = int((hc.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT).sum())
print(f"\nmalignant CD4 in HC skin: {n_mal} (expected 0)")
assert n_mal == 0, "a malignant CD4 call in healthy skin needs explaining before going further"

hc_bal, _ = C.subsample_levels(hc, verbose=False)
res_hc = C.run_rank_aggregate(hc_bal, resource, groupby_pairs=pairs_hc,
                              key_added="liana_hc", verbose=False)
res_hc.to_csv(cd.TAB_DIR / "ccc_liana_hc_structural.csv", index=False)
display(res_hc.sort_values("magnitude_rank").head(20))
```

```
window: 23410 cells, 9 donors, 9 samples, 3 studies
donors per level:
ccc_celltype
CD8               9
CD4_unassessed    9
Myeloid           9
Melanocyte        9
Vascular          9
Fibroblast        8
Keratinocyte      8
Mast              6
Plasma            6
B                 5
CD4_reactive      3
Tregs             2

hc_structural: ['Fibroblast'] <-> ['Keratinocyte', 'Vascular', 'Myeloid']
  Negative control. HC skin has 27,534 cells but only 1,524 CD4 and 300 CD8 across 9 donors, so no T-cell axis is computable -- only the structural fibroblast -> keratinocyte/vascular/myeloid grid. CD4_malignant must be 0 cells here; that is the expected result.

malignant CD4 in HC skin: 0 (expected 0)

          source        target ligand_complex receptor_complex  lr_means  \
343   Fibroblast       Myeloid            APP             CD74  3.313453   
1545    Vascular    Fibroblast            VIM             CD44  2.839261   
591   Fibroblast       Myeloid            VIM             CD44  2.572871   
315   Fibroblast  Keratinocyte            VIM             CD44  2.536706   
396   Fibroblast       Myeloid         COL1A2             CD44  2.530972   
75    Fibroblast  Keratinocyte         COL1A2             CD44  2.494807   
392   Fibroblast       Myeloid         COL1A1             CD44  2.400812   
69    Fibroblast  Keratinocyte         COL1A1             CD44  2.364647   
631   Fibroblast      Vascular            APP             CD74  2.354650   
353   Fibroblast       Myeloid            B2M             CD1C  2.456227   
1285     Myeloid    Fibroblast            VIM             CD44  2.282142   
1264     Myeloid    Fibroblast          TIMP1             CD63  2.213743   
854   Fibroblast      Vascular           MMP2           PECAM1  2.202934   
489   Fibroblast       Myeloid         LGALS1            PTPRC  2.109277   
572   Fibroblast       Myeloid          TIMP1             CD63  2.092055   
918   Fibroblast      Vascular          TIMP1             CD63  2.081788   
835   Fibroblast      Vascular         LGALS1            ITGB1  2.075032   
404   Fibroblast       Myeloid         COL6A2             CD44  2.042604   
1420    Vascular    Fibroblast          HLA-A            APLP2  2.041224   
1538    Vascular    Fibroblast          TIMP1             CD63  2.021207   

      cellphone_pvals  expr_prod  scaled_weight  lr_logfc  spec_weight  \
343               0.0   7.068313       1.005924  1.943336     0.592067   
1545              0.0   5.515693       0.506658  0.941839     0.541147   
591               0.0   5.018434       0.375130  0.267288     0.477145   
315               0.0   4.740812       0.339741  0.411312     0.450749   
396               0.0   4.908870       0.745686  2.960358     0.477145   
75                0.0   4.637309       0.710297  3.104381     0.450749   
392               0.0   4.568509       0.717502  2.960244     0.477145   
69                0.0   4.315777       0.682113  3.104267     0.450749   
631               0.0   4.506556       0.550366  0.928980     0.377485   
353               0.0   3.675360       1.075174  0.875252     0.996281   
1285              0.0   4.129879       0.202722 -0.006935     0.405184   
1264              0.0   4.551860       0.442218  0.949510     0.533079   
854               0.0   4.834167       1.110327  3.769120     0.898276   
489               0.0   3.179574       1.704205  2.294142     0.976368   
572               0.0   4.278262       0.323083  0.744224     0.417152   
918               0.0   4.228861       0.314520  0.618003     0.412335   
835               0.0   2.957937       0.533964  1.423850     0.526549   
404               0.0   3.631815       0.713940  2.170385     0.477145   
1420              0.0   2.932851       0.434694  0.840047     0.449756   
1538              0.0   3.471986       0.290291  0.386903     0.406612   

       lrscore  specificity_rank  magnitude_rank  
343   0.950688          0.073200        0.000004  
1545  0.944538          0.221011        0.
... [924 more chars truncated]
```

### Outcome

Result frames are in `tables/ccc_*.csv`. Read the reportable set as the intersection of:
top-ranked in §2/§3 · shifted against the reactive comparator in §6 · present in the control
report §7 · stable across `expr_prop` §8 · dissolved by the within-donor shuffle §9 · stable
across ≥3 malignancy definitions §13.

Headline figures: **`36_ccc_headline_figures.ipynb`**.
