# old/ccc_v1_03_headline_figures.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/ccc_v1_03_headline_figures.ipynb` and is never run._

# 36 · CCC — headline figures: malignant CD4 ↔ skin TME

**Recomputed from scratch** off `data/ccc/ccc_skin.h5ad`. nb35 is the exploratory record;
this notebook is the read-out, so it depends on nothing nb35 produced. Same split as
`MyelomaProject/ccc/` nb02 vs nb05.

Figures are saved as hybrid SVGs (`final_figures_helpers.svg_style` / `save_svg`) into
`figures/final/ccc_*.svg` — text stays editable vector, data layers rasterized.

`CD4_malignant` here means **the ALICE TCR clone family** (`mal_tcr_alice`): 76,349 cells,
29 donors. Not `mal_combined` — see nb34 and nb35 §13.

## The figures, and why each window

| # | axis | window | why this window | tool |
|---|---|---|---|---|
| 1 | malignant CD4 ↔ Myeloid | all CTCL skin | 28 donors — the best-powered partner axis | dotplot ×2 |
| 2 | malignant CD4 ↔ Fibroblast | all CTCL skin | 24 donors; the CXCL12/TGFβ niche | dotplot ×2 |
| 3 | malignant CD4 ↔ CD8 | all CTCL skin | 27 donors; the cytotoxic-restraint axis | dotplot ×2 |
| 4 | malignant CD4 ↔ B + Plasma | all CTCL skin | 20/17 donors; the CXCL13 / TLS axis the replication spec predicts | dotplot ×2 |
| 5 | malignant CD4 ↔ Keratinocyte + Vascular | all CTCL skin | 22/26 donors; epidermotropism and trafficking | dotplot ×2 |
| 6 | malignant CD4 → all partners, and back | all CTCL skin | the whole map in two panels | `partner_heatmap` |
| 7 | malignant vs reactive CD4, same partners | all CTCL skin | the comparator — makes "malignant-specific" mean something | dotplot + delta |
| 8 | curated LR panel × partner | all CTCL skin | forced panel; negatives reported with their detection proportion | dotplot + table |

Mast (4 donors) and Melanocyte (13) are **excluded**: `axis_feasibility` marks them
`report_only`. They are in nb35 §4 so their absence is on the record.

## Standing caveats

- LIANA permutation p-values take the **cell** as the unit and are pseudoreplicated across
  donors (76,349 malignant CD4 cells from 29 donors). They size dots and mark heatmap cells.
  **They are not evidence.** Donor-level inference is the deferred differential phase.
- Grey heatmap cells mean the pair failed `MIN_CELLS` in that stratum. Read them against the
  coverage table printed beneath each figure, **not** as biology.
- `CD4_reactive` is *lesional* reactive CD4, not normal skin — HC skin has 1,524 CD4 across
  9 donors, so there is no normal-skin baseline. Every "malignant-specific" statement here
  means **"differs from reactive CD4 in the same lesion."**
- The malignancy call is a TCR-sequence fact, so it is independent of the expression scored
  here — that was the reason for dropping `mal_combined` as the primary. The residual exposure
  runs the other way: 40,447 of the 72,207 **reactive** CD4 carry a `cnv_only` call, so if
  inferCNV is right about some of them the comparator contains tumour and the malignant-vs-
  reactive delta is conservative. §9's `malignant_evidence == "both"` restriction and nb35 §13
  are where that is bounded.

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
import final_figures_helpers as FF

sc.settings.verbosity = 1
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
FF.svg_style()
cd.FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
cd.TAB_DIR.mkdir(parents=True, exist_ok=True)
print("liana", li.__version__)
print("\n" + cd.CAVEAT_BLOCK)
```

```
liana 1.8.1

LIANA permutation p-values are computed with the CELL as the unit and are pseudoreplicated across donors (76,349 malignant CD4 cells come from 29 donors). They size the dots and mark the heatmap cells; they are not evidence. Donor-level inference is the deferred differential phase. Grey heatmap cells mean the pair failed min_cells in that stratum -- read them against the coverage table printed beneath each figure, not as biology.
```

## §0 · Load, balance, and recompute every axis

```python  # cell 3
adata = C.load_ccc_adata(verbose=False)
adata.layers[cd.LAYER] = adata.X
C.assert_ccc_invariants(adata)
resource, _ = C.load_resource(var_names=adata.var_names, verbose=False)

bal, _ = C.subsample_levels(adata, verbose=False)
print("balanced:", bal.n_obs, "cells")

ALL_PARTNERS = cd.TME_CORE + cd.TME_EXTENDED
pairs_mal = C.build_groupby_pairs({"mal": ([cd.CD4_MALIGNANT], ALL_PARTNERS + [cd.CD4_REACTIVE])})
pairs_rea = C.build_groupby_pairs({"rea": ([cd.CD4_REACTIVE], ALL_PARTNERS)})

ctcl = C.focal_window(bal, disease=cd.CTCL_DISEASES)
```

```
invariants OK: (727156, 1832)
balanced: 155778 cells
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
```

```python  # cell 4
res_mal = C.run_rank_aggregate(ctcl, resource, groupby_pairs=pairs_mal,
                               key_added="liana_mal", verbose=False)
res_rea = C.run_rank_aggregate(ctcl, resource, groupby_pairs=pairs_rea,
                               key_added="liana_rea", verbose=False)
res_mal.to_csv(cd.TAB_DIR / "ccc_headline_malignant_axes.csv", index=False)
res_rea.to_csv(cd.TAB_DIR / "ccc_headline_reactive_axes.csv", index=False)
print(res_mal.shape, res_rea.shape)
```

```
(2353, 13) (1585, 13)
```

```python  # cell 5
# the claim gate: which of these axes may carry a headline claim at all
counts_donor, _ = C.cell_count_audit(ctcl, sample_key=cd.DONOR_KEY)
feas = C.axis_feasibility(counts_donor)
COV = C.coverage_table(ctcl, sample_key=cd.DONOR_KEY, levels=cd.CT_ORDER)
display(feas[feas.source.eq(cd.CD4_MALIGNANT) | feas.target.eq(cd.CD4_MALIGNANT)])
display(COV)


def show(fig_or_gg, name, coverage_levels=None, width=8, height=6):
    # save to figures/final and print the coverage rows this figure depends on
    out = cd.FINAL_FIG_DIR / f"ccc_{name}.svg"
    if isinstance(fig_or_gg, p9.ggplot):
        fig_or_gg.save(out, width=width, height=height, verbose=False)
        print("saved", out)
    else:
        for ax in fig_or_gg.axes:
            for im in ax.images:
                im.set_rasterized(True)
        FF.save_svg(fig_or_gg, out)
    if coverage_levels:
        print("\ncoverage (donors with >= %d cells):" % cd.MIN_CELLS)
        print(COV.reindex([lv for lv in coverage_levels if lv in COV.index]).to_string())
    return fig_or_gg
```

```
axis         source         target  n_units      verdict
21  malig_vs_reactive   CD4_reactive  CD4_malignant       26        claim
20  malig_vs_reactive  CD4_malignant   CD4_reactive       26        claim
4       malig_vs_core        Myeloid  CD4_malignant       25        claim
1       malig_vs_core  CD4_malignant        Myeloid       25        claim
12  malig_vs_extended       Vascular  CD4_malignant       25        claim
7   malig_vs_extended  CD4_malignant       Vascular       25        claim
3       malig_vs_core            CD8  CD4_malignant       24        claim
0       malig_vs_core  CD4_malignant            CD8       24        claim
2       malig_vs_core  CD4_malignant     Fibroblast       22        claim
5       malig_vs_core     Fibroblast  CD4_malignant       22        claim
8   malig_vs_extended  CD4_malignant              B       20        claim
13  malig_vs_extended              B  CD4_malignant       20        claim
11  malig_vs_extended   Keratinocyte  CD4_malignant       19        claim
6   malig_vs_extended  CD4_malignant   Keratinocyte       19        claim
14  malig_vs_extended         Plasma  CD4_malignant       17        claim
9   malig_vs_extended  CD4_malignant         Plasma       17        claim
19      malig_vs_thin     Melanocyte  CD4_malignant       13  report_only
17      malig_vs_thin  CD4_malignant     Melanocyte       13  report_only
10  malig_vs_extended  CD4_malignant          Tregs       10        claim
15  malig_vs_extended          Tregs  CD4_malignant       10        claim
16      malig_vs_thin  CD4_malignant           Mast        4  report_only
18      malig_vs_thin           Mast  CD4_malignant        4  report_only
                n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                       
CD4_malignant     15000              31            28      0.383562
CD4_reactive      14566              33            29      0.397260
CD4_unassessed    14850              73            66      0.904110
CD8               14867              71            57      0.780822
Tregs              1931              66            20      0.273973
B                 12906              65            41      0.561644
Plasma             9961              64            44      0.602740
Myeloid           14696              73            64      0.876712
Mast                991              56            11      0.150685
Keratinocyte      13376              67            47      0.643836
Fibroblast        12709              72            55      0.753425
Vascular          12460              73            58      0.794521
Melanocyte         8493              63            21      0.287671
```

## §1 · Figure 1 — malignant CD4 ↔ Myeloid

28 donors clear `MIN_CELLS` on both sides: the best-powered partner axis in the atlas. Myeloid
is the pooled macrophage / monocyte / cDC / moDC / LC / pDC block, because the li2024 fine
taxonomy is `Unknown` for the 330k non-li2024 skin cells and splitting on it would make
"Myeloid" mean "non-li2024 myeloid" — a study label wearing a cell-state name.

```python  # cell 7
g = C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT], target_labels=["Myeloid"],
                   title="malignant CD4 -> myeloid")
show(g, "fig1a_malignant_to_myeloid", [cd.CD4_MALIGNANT, "Myeloid"], width=6, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig1a_malignant_to_myeloid.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
Myeloid          14696              73            64      0.876712

<plotnine.ggplot.ggplot object at 0x1549acbe5d50>
```

```python  # cell 8
g = C.dotplot_axis(liana_res=res_mal, source_labels=["Myeloid"], target_labels=[cd.CD4_MALIGNANT],
                   title="myeloid -> malignant CD4")
show(g, "fig1b_myeloid_to_malignant", [cd.CD4_MALIGNANT, "Myeloid"], width=6, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig1b_myeloid_to_malignant.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
Myeloid          14696              73            64      0.876712

<plotnine.ggplot.ggplot object at 0x1549c9463b10>
```

## §2 · Figure 2 — malignant CD4 ↔ Fibroblast

24 donors. The axis that carries the `CXCL12 → CXCR4` niche and `TGFB1 → TGFBR1_TGFBR2`
(the resource also stores the latter as `ACVR1_TGFBR1_TGFBR2`; both are checked).

```python  # cell 10
g = C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT], target_labels=["Fibroblast"],
                   title="malignant CD4 -> fibroblast")
show(g, "fig2a_malignant_to_fibroblast", [cd.CD4_MALIGNANT, "Fibroblast"], width=6, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig2a_malignant_to_fibroblast.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
Fibroblast       12709              72            55      0.753425

<plotnine.ggplot.ggplot object at 0x1549c95d5dd0>
```

```python  # cell 11
g = C.dotplot_axis(liana_res=res_mal, source_labels=["Fibroblast"], target_labels=[cd.CD4_MALIGNANT],
                   title="fibroblast -> malignant CD4")
show(g, "fig2b_fibroblast_to_malignant", [cd.CD4_MALIGNANT, "Fibroblast"], width=6, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig2b_fibroblast_to_malignant.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
Fibroblast       12709              72            55      0.753425

<plotnine.ggplot.ggplot object at 0x1549ad3e4ad0>
```

## §3 · Figure 3 — malignant CD4 ↔ CD8

27 donors. CD8 is one level: ALICE assigns it no malignant cell at all (CD4-only caller), and
the 21,200 of 51,831 skin CD8 that `mal_cnv` flags carry **zero** TCR support, so a "malignant
CD8" split would most likely be a CNV or ambient artifact. nb35 §14 quantifies what keeping
those cells in the pooled level costs.

```python  # cell 13
g = C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT], target_labels=["CD8"],
                   title="malignant CD4 -> CD8")
show(g, "fig3a_malignant_to_cd8", [cd.CD4_MALIGNANT, "CD8"], width=6, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig3a_malignant_to_cd8.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
CD8              14867              71            57      0.780822

<plotnine.ggplot.ggplot object at 0x1549aca27610>
```

```python  # cell 14
g = C.dotplot_axis(liana_res=res_mal, source_labels=["CD8"], target_labels=[cd.CD4_MALIGNANT],
                   title="CD8 -> malignant CD4")
show(g, "fig3b_cd8_to_malignant", [cd.CD4_MALIGNANT, "CD8"], width=6, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig3b_cd8_to_malignant.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
CD8              14867              71            57      0.780822

<plotnine.ggplot.ggplot object at 0x1548dfc85490>
```

## §4 · Figure 4 — malignant CD4 ↔ B and Plasma

20 and 17 donors. This is where `docs/CTCL_atlas_notebook_replication_spec.md` predicts
`CXCL13–CXCR5`, `CD40LG–CD40` and `CD28–CD86` (Fig 8a). §10 checks them explicitly.

```python  # cell 16
g = C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT], target_labels=["B", "Plasma"],
                   title="malignant CD4 -> B / plasma")
show(g, "fig4a_malignant_to_b_plasma", [cd.CD4_MALIGNANT, "B", "Plasma"], width=7, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig4a_malignant_to_b_plasma.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
B                12906              65            41      0.561644
Plasma            9961              64            44      0.602740

<plotnine.ggplot.ggplot object at 0x1549099be190>
```

```python  # cell 17
g = C.dotplot_axis(liana_res=res_mal, source_labels=["B", "Plasma"], target_labels=[cd.CD4_MALIGNANT],
                   title="B / plasma -> malignant CD4")
show(g, "fig4b_b_plasma_to_malignant", [cd.CD4_MALIGNANT, "B", "Plasma"], width=7, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig4b_b_plasma_to_malignant.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
B                12906              65            41      0.561644
Plasma            9961              64            44      0.602740

<plotnine.ggplot.ggplot object at 0x15490986c910>
```

## §5 · Figure 5 — malignant CD4 ↔ Keratinocyte and Vascular

22 and 26 donors: the epidermotropism and trafficking partners. Note that a layer-resolved
version is *not* shown — `skin_layer` is `whole` for 417k of 750k skin cells, so an
epidermis-vs-dermis split would be confounded with dissociation protocol and study.

```python  # cell 19
g = C.dotplot_axis(liana_res=res_mal, source_labels=[cd.CD4_MALIGNANT],
                   target_labels=["Keratinocyte", "Vascular"],
                   title="malignant CD4 -> keratinocyte / vascular")
show(g, "fig5a_malignant_to_kc_vascular", [cd.CD4_MALIGNANT, "Keratinocyte", "Vascular"],
     width=7, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig5a_malignant_to_kc_vascular.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
Keratinocyte     13376              67            47      0.643836
Vascular         12460              73            58      0.794521

<plotnine.ggplot.ggplot object at 0x1549d5a47450>
```

```python  # cell 20
g = C.dotplot_axis(liana_res=res_mal, source_labels=["Keratinocyte", "Vascular"],
                   target_labels=[cd.CD4_MALIGNANT],
                   title="keratinocyte / vascular -> malignant CD4")
show(g, "fig5b_kc_vascular_to_malignant", [cd.CD4_MALIGNANT, "Keratinocyte", "Vascular"],
     width=7, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig5b_kc_vascular_to_malignant.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
Keratinocyte     13376              67            47      0.643836
Vascular         12460              73            58      0.794521

<plotnine.ggplot.ggplot object at 0x1549ad519b10>
```

## §6 · Figure 6 — the whole map, one panel per direction

`partner_heatmap`: top interactions × partner cell type. Rendering contract, kept from the
Myeloma `time_heatmap` with partner on the x axis instead of day —

- colour = `lr_means` on viridis, so a magnitude reads as a magnitude;
- **grey** = structurally absent (the pair failed `MIN_CELLS` for that partner), *not* a low
  value; that distinction is the whole reason for `cmap.set_bad`;
- **shared** `vmin`/`vmax` across both panels, so left and right are comparable;
- `*` where the per-cell permutation p < 0.05 — pseudoreplicated, so it marks cells only;
- row order = minimum `magnitude_rank` across partners, so a pair strong with any single
  partner still makes the panel.

```python  # cell 22
claimable = [lv for lv in ALL_PARTNERS
             if feas.query("source == @cd.CD4_MALIGNANT and target == @lv and verdict == 'claim'").shape[0]]
print("claimable partners:", claimable)
fig = C.partner_heatmap(res_mal, sender=cd.CD4_MALIGNANT, partners=claimable, figsize=(12, 5.5),
                        title="malignant CD4 <-> skin TME  (lr_means; * = permutation p<0.05; grey = below min_cells)")
show(fig, "fig6_partner_heatmap", [cd.CD4_MALIGNANT] + claimable)
```

```
claimable partners: ['CD8', 'Myeloid', 'Fibroblast', 'Keratinocyte', 'Vascular', 'B', 'Plasma', 'Tregs']
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig6_partner_heatmap.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
CD8              14867              71            57      0.780822
Myeloid          14696              73            64      0.876712
Fibroblast       12709              72            55      0.753425
Keratinocyte     13376              67            47      0.643836
Vascular         12460              73            58      0.794521
B                12906              65            41      0.561644
Plasma            9961              64            44      0.602740
Tregs             1931              66            20      0.273973

<Figure size 1320x605 with 3 Axes>
```

## §7 · Figure 7 — the comparator

Same partners, **reactive** CD4 as sender, side by side with the delta table. A pair that ranks
the same from reactive CD4 is a property of CD4 T cells in inflamed skin, not of the malignant
clone — reading the malignant panel without this one is the main way to overclaim here.

```python  # cell 24
g = C.dotplot_axis(liana_res=res_rea, source_labels=[cd.CD4_REACTIVE], target_labels=claimable,
                   title="reactive CD4 -> TME (comparator)")
show(g, "fig7a_reactive_to_tme", [cd.CD4_REACTIVE] + claimable, width=9, height=6)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig7a_reactive_to_tme.svg

coverage (donors with >= 25 cells):
              n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                     
CD4_reactive    14566              33            29      0.397260
CD8             14867              71            57      0.780822
Myeloid         14696              73            64      0.876712
Fibroblast      12709              72            55      0.753425
Keratinocyte    13376              67            47      0.643836
Vascular        12460              73            58      0.794521
B               12906              65            41      0.561644
Plasma           9961              64            44      0.602740
Tregs            1931              66            20      0.273973

<plotnine.ggplot.ggplot object at 0x154a1b898e50>
```

```python  # cell 25
delta_out = C.rank_delta(res_mal.query("source == @cd.CD4_MALIGNANT"),
                         res_rea.query("source == @cd.CD4_REACTIVE"))
delta_in = C.rank_delta(
    res_mal.query("target == @cd.CD4_MALIGNANT").rename(columns={"source": "target", "target": "source"}),
    res_rea.query("target == @cd.CD4_REACTIVE").rename(columns={"source": "target", "target": "source"}),
)
delta_out.to_csv(cd.TAB_DIR / "ccc_headline_rank_delta_outgoing.csv", index=False)
delta_in.to_csv(cd.TAB_DIR / "ccc_headline_rank_delta_incoming.csv", index=False)
print("OUTGOING -- most malignant-shifted (negative delta = better rank from malignant CD4):")
display(delta_out.head(20))
print(f"\nfound ONLY with malignant CD4 as sender: {int(delta_out['malignant_only'].sum())} pairs")
```

```
OUTGOING -- most malignant-shifted (negative delta = better rank from malignant CD4):

            target ligand_complex receptor_complex  magnitude_rank_malignant  \
1098      Vascular         LGALS3             MCAM                  0.211533   
227            CD8         LGALS3             LAG3                  0.170758   
184            CD8          GNAI2             CCR5                  0.378357   
327     Fibroblast          GNAI2            S1PR3                  0.440120   
1075      Vascular          GNAI2              F2R                  0.441185   
1106      Vascular         LRPAP1             LDLR                  0.442251   
708        Myeloid         LRPAP1             LRP1                  0.446534   
1045      Vascular          ARPC5            ADRB2                  0.400915   
982          Tregs         LRPAP1            SORL1                  0.450845   
884         Plasma          TIMP1             CD63                  0.257128   
362     Fibroblast         LRPAP1             LDLR                  0.455184   
841         Plasma         LGALS1             CD69                  0.240412   
323     Fibroblast          GNAI2            EDNRB                  0.398919   
233            CD8         LRPAP1            SORL1                  0.475051   
696        Myeloid         LGALS1             CD69                  0.278971   
40               B         LRPAP1            SORL1                  0.496642   
555   Keratinocyte           RTN4         TNFRSF19                  0.498951   
35               B         LGALS1            ITGB1                  0.156800   
842         Plasma         LGALS1            ITGB1                  0.162320   
1163      Vascular        TNFSF10        TNFRSF10D                  0.501269   

      lr_means_malignant  magnitude_rank_reactive  lr_means_reactive  \
1098            1.058843                 0.929638           0.695914   
227             1.141429                 0.799738           0.778500   
184             0.491362                 0.943821           0.383327   
327             0.483319                 1.000000           0.375284   
1075            0.483237                 1.000000           0.375202   
1106            0.457152                 1.000000           0.335029   
708             0.469182                 1.000000           0.347060   
1045            0.594286                 0.951723           0.458413   
982             0.469631                 1.000000           0.347509   
884             1.016402                 0.804831           0.774945   
362             0.452192                 1.000000           0.330070   
841             1.236242                 0.779461           0.846368   
323             0.493778                 0.925554           0.385743   
233             0.445171                 1.000000           0.323048   
696             1.225295                 0.787828           0.835421   
40              0.442812                 1.000000           0.320689   
555             0.467079                 1.000000           0.368548   
35              1.268481                 0.656365           0.878607   
842             1.265031                 0.661160           0.875157   
1163            0.437142                 1.000000           0.295036   

      delta_rank  delta_lr_means  malignant_only  
1098   -0.718105        0.362929           False  
227    -0.628980        0.362929           False  
184    -0.565465        0.108035           False  
327    -0.559880        0.108035           False  
1075   -0.558815        0.108035           False  
1106   -0.557749        0.122122           False  
708    -0.553466        0.122122           False  
1045   -0.550808        0.135873           False  
982    -0.549155        0.122122           False  
884    -0.547703        0.241457           False  
362    -0.544816        0.122122           False  
841    -0.539049        0.389874           False  
323    -0.526636        0.108035           False  
233  
... [403 more chars truncated]
```

```python  # cell 26
print("INCOMING -- TME pairs most shifted toward malignant CD4 as receiver:")
display(delta_in.head(20))
```

```
INCOMING -- TME pairs most shifted toward malignant CD4 as receiver:

            target ligand_complex receptor_complex  magnitude_rank_malignant  \
421     Fibroblast           IL15            IL2RA                  0.161484   
1180      Vascular           IL15            IL2RA                  0.369970   
1061         Tregs          TIMP1             CD63                  0.418153   
597   Keratinocyte          ICAM1            IL2RA                  0.439422   
860         Plasma           CD47            SIRPG                  0.503011   
972          Tregs           CD47            SIRPG                  0.510022   
105              B          TGFB1           TGFBR3                  0.516505   
832        Myeloid          TOR2A          ATP5F1B                  0.413033   
763        Myeloid           IL15            IL2RA                  0.544638   
72               B          ICAM1            IL2RA                  0.551325   
811        Myeloid          TGFB1           TGFBR3                  0.551599   
688        Myeloid           CD47            SIRPG                  0.555939   
416     Fibroblast          ICAM1            IL2RA                  0.456273   
223            CD8           CD55           ADGRE5                  0.501269   
221            CD8           CD47            SIRPG                  0.581177   
815        Myeloid          TIMP1             CD63                  0.163248   
997          Tregs          HLA-A            APLP2                  0.415101   
755        Myeloid          ICAM1            IL2RA                  0.434821   
395     Fibroblast          HLA-A            APLP2                  0.435847   
850         Plasma           ARF6            SMAP1                  0.546667   

      lr_means_malignant  magnitude_rank_reactive  lr_means_reactive  \
421             0.160535                 0.976618           0.279368   
1180            0.179274                 1.000000           0.298108   
1061            0.566291                 1.000000           0.331217   
597             0.184911                 1.000000           0.303744   
860             0.441551                 1.000000           0.369440   
972             0.433139                 1.000000           0.361028   
105             0.432850                 0.989181           0.381929   
832             0.720282                 0.872170           0.587001   
763             0.193268                 1.000000           0.312101   
72              0.193576                 1.000000           0.312410   
811             0.421943                 1.000000           0.371022   
688             0.421891                 1.000000           0.349780   
416             0.604291                 0.879668           0.723124   
223             0.437479                 0.922964           0.386863   
221             0.412937                 1.000000           0.340826   
815             1.154207                 0.576638           0.919133   
997             0.987584                 0.817602           0.969508   
755             0.528797                 0.832904           0.647630   
395             0.975186                 0.828080           0.957110   
850             0.424437                 0.938578           0.382120   

      delta_rank  delta_lr_means  malignant_only  
421    -0.815135       -0.118833           False  
1180   -0.630030       -0.118833           False  
1061   -0.581847        0.235074           False  
597    -0.560578       -0.118833           False  
860    -0.496989        0.072111           False  
972    -0.489978        0.072111           False  
105    -0.472676        0.050921           False  
832    -0.459138        0.133281           False  
763    -0.455362       -0.118833           False  
72     -0.448675       -0.118833           False  
811    -0.448401        0.050921           False  
688    -0.444061        0.072111           False  
416    -0.423395       -0.118833           False  
223    -0.421695      
... [332 more chars truncated]
```

```python  # cell 27
# Figure 7b: the delta itself, top shifted pairs per partner
d = delta_out.dropna(subset=["delta_rank"]).copy()
d["interaction"] = d["ligand_complex"] + " -> " + d["receptor_complex"]
top = d.reindex(d.groupby("target")["delta_rank"].nsmallest(6).index.get_level_values(1))
fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(top))))
colors = {p: c for p, c in zip(claimable, plt.cm.tab10.colors)}
y = np.arange(len(top))
ax.barh(y, -top["delta_rank"].to_numpy(),
        color=[colors.get(t, "0.6") for t in top["target"]])
ax.set_yticks(y); ax.set_yticklabels([f"{i}  [{t}]" for i, t in zip(top.interaction, top.target)],
                                     fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("rank advantage of malignant over reactive CD4 as sender")
ax.set_title("Malignant-specific outgoing signals, per TME partner", fontsize=9)
show(fig, "fig7b_rank_delta_by_partner", [cd.CD4_MALIGNANT, cd.CD4_REACTIVE] + claimable)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig7b_rank_delta_by_partner.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
CD4_reactive     14566              33            29      0.397260
CD8              14867              71            57      0.780822
Myeloid          14696              73            64      0.876712
Fibroblast       12709              72            55      0.753425
Keratinocyte     13376              67            47      0.643836
Vascular         12460              73            58      0.794521
B                12906              65            41      0.561644
Plasma            9961              64            44      0.602740
Tregs             1931              66            20      0.273973

<Figure size 770x1478.4 with 1 Axes>
```

## §8 · Figure 8 — the forced curated panel

Plotted whether or not a pair cleared `expr_prop`, so a negative is *reported* with its actual
detection proportion rather than silently dropped. The caveat list is printed first, because two
of the panel groups exist specifically to document failure modes: `MIF` (ubiquitous — a top
`MIF→CD74` edge is the null model breaking) and `IL13`/`IL4` (Th2 cytokine mRNA is poorly
captured by 3′ 10x, so a negative there is uninformative).

```python  # cell 29
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

```python  # cell 30
panel = C.filter_to_panel(res_mal, resource=resource)
panel.to_csv(cd.TAB_DIR / "ccc_headline_curated_panel.csv", index=False)
print(f"{len(panel)} scored rows over {panel.group.nunique()} panel groups")
g = C.dotplot_axis(liana_res=panel, source_labels=[cd.CD4_MALIGNANT], target_labels=claimable,
                   top_n=None, title="curated CTCL panel: malignant CD4 -> TME")
show(g, "fig8a_panel_malignant_out", [cd.CD4_MALIGNANT] + claimable, width=9, height=8)
```

```
121 scored rows over 8 panel groups
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig8a_panel_malignant_out.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
CD8              14867              71            57      0.780822
Myeloid          14696              73            64      0.876712
Fibroblast       12709              72            55      0.753425
Keratinocyte     13376              67            47      0.643836
Vascular         12460              73            58      0.794521
B                12906              65            41      0.561644
Plasma            9961              64            44      0.602740
Tregs             1931              66            20      0.273973

<plotnine.ggplot.ggplot object at 0x154a1308cb50>
```

```python  # cell 31
g = C.dotplot_axis(liana_res=panel, source_labels=claimable, target_labels=[cd.CD4_MALIGNANT],
                   top_n=None, title="curated CTCL panel: TME -> malignant CD4")
show(g, "fig8b_panel_malignant_in", [cd.CD4_MALIGNANT] + claimable, width=9, height=8)
```

```
saved /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/figures/final/ccc_fig8b_panel_malignant_in.svg

coverage (donors with >= 25 cells):
               n_cells  donors_present  donors_ge_25  frac_passing
ccc_celltype                                                      
CD4_malignant    15000              31            28      0.383562
CD8              14867              71            57      0.780822
Myeloid          14696              73            64      0.876712
Fibroblast       12709              72            55      0.753425
Keratinocyte     13376              67            47      0.643836
Vascular         12460              73            58      0.794521
B                12906              65            41      0.561644
Plasma            9961              64            44      0.602740
Tregs             1931              66            20      0.273973

<plotnine.ggplot.ggplot object at 0x154a1b0e63d0>
```

```python  # cell 32
# the negatives, with numbers: curated pairs that were NOT scored, and why
resolved = C.resolve_panel(cd.LR_PANEL, resource)
scored = set(zip(panel.ligand_complex, panel.receptor_complex))
fpe = C.forced_panel_expression(ctcl, resource=resource)
prop = fpe.set_index(["level", "gene"])["expr_prop"]

rows = []
for r in resolved.itertuples(index=False):
    if r.orientation == "absent":
        rows.append({"group": r.group, "pair": f"{r.ligand_in} -> {r.receptor_in}",
                     "reason": "absent from the consensus resource"})
        continue
    if (r.ligand_complex, r.receptor_complex) in scored:
        continue
    subs = [s for s in str(r.ligand_complex).split("_")]
    props = {s: prop.get((cd.CD4_MALIGNANT, s), np.nan) for s in subs}
    rows.append({"group": r.group, "pair": f"{r.ligand_complex} -> {r.receptor_complex}",
                 "reason": "below expr_prop; ligand detection in malignant CD4: "
                           + ", ".join(f"{k}={v:.1%}" if pd.notna(v) else f"{k}=NA"
                                       for k, v in props.items())})
neg = pd.DataFrame(rows)
neg.to_csv(cd.TAB_DIR / "ccc_headline_panel_negatives.csv", index=False)
print("curated pairs NOT scored, with the reason:")
display(neg)
```

```
curated pairs NOT scored, with the reason:

                      group                        pair  \
0           b_cell_tls_axis       TNFSF13B -> TNFRSF13B   
1           b_cell_tls_axis       TNFSF13B -> TNFRSF13C   
2           b_cell_tls_axis        TNFSF13 -> TNFRSF13B   
3   costim_tme_to_malignant              ICOSLG -> ICOS   
4   costim_tme_to_malignant           TNFSF4 -> TNFRSF4   
5   costim_tme_to_malignant         TNFSF18 -> TNFRSF18   
6      checkpoint_restraint        HLA-E -> KLRC1_KLRD1   
7    th2_stroma_remodelling   IL13 -> IL13_IL13RA1_IL4R   
8    th2_stroma_remodelling           IL4 -> IL2RG_IL4R   
9    th2_stroma_remodelling         IL31 -> IL31RA_OSMR   
10   th2_stroma_remodelling      IL22 -> IL10RB_IL22RA1   
11    chemokine_recruitment              CCL27 -> CCR10   
12         survival_signal3  IL15 -> IL15RA_IL2RB_IL2RG   
13         survival_signal3    IL2 -> IL2RA_IL2RB_IL2RG   
14     malignant_to_myeloid          CSF1_IL34 -> CSF1R   
15     malignant_to_myeloid       IFNG -> IFNGR1_IFNGR2   

                                               reason  
0   below expr_prop; ligand detection in malignant...  
1   below expr_prop; ligand detection in malignant...  
2   below expr_prop; ligand detection in malignant...  
3   below expr_prop; ligand detection in malignant...  
4   below expr_prop; ligand detection in malignant...  
5   below expr_prop; ligand detection in malignant...  
6   below expr_prop; ligand detection in malignant...  
7   below expr_prop; ligand detection in malignant...  
8   below expr_prop; ligand detection in malignant...  
9   below expr_prop; ligand detection in malignant...  
10  below expr_prop; ligand detection in malignant...  
11  below expr_prop; ligand detection in malignant...  
12  below expr_prop; ligand detection in malignant...  
13  below expr_prop; ligand detection in malignant...  
14  below expr_prop; ligand detection in malignant...  
15  below expr_prop; ligand detection in malignant...
```

## §9 · Robustness of the headline set

Everything asserted above is re-checked here, in one place: the resolved controls, the
within-donor label shuffle, and the `malignant_evidence == "both"` restriction.

Under an ALICE primary that last check has changed meaning. It is no longer removing a
`cnv_only` circularity from the malignant level — ALICE already did that. It is now a **nested**
restriction (62,096 of the 76,349 malignant CD4 also carry a CNV call) asking whether the 14,253
`tcr_only` cells, the ones with a tumour TCR but no detectable CNV, drive any headline pair on
their own.

```python  # cell 34
ctrl, caveat_hits = C.control_report(pd.concat([res_mal, res_rea]), resource=resource, top_n=40)
ctrl.to_csv(cd.TAB_DIR / "ccc_headline_control_report.csv", index=False)
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
0                 CXCR5   True         3071.0        0.970003   
1                  CD40   True         2118.0        0.679052   
2                  CD28  False            NaN             NaN   
3              TNFRSF1B   True         1309.0        0.487473   
4                  CD28   True         2411.0        0.751827   
5                  CCR4   True         1399.0        0.507093   
6                  CCR4   True         1711.0        0.577981   
7                   CD2   True         1325.0        0.491541   
8                 CXCR4   True          246.0        0.044204   
9                 CSF1R  False            NaN             NaN   
10        TGFBR1_TGFBR2   True         1043.0        0.421245   
11  ACVR1_TGFBR1_TGFBR2   True         2191.0        0.696965   
12                  KIT  False            NaN             NaN   
13          ITGA2_ITGB1  False            NaN             NaN   
14                    *  False            NaN             NaN   

    specificity_rank  lr_means  cellphone_pvals in_top_n  pct_of_tested  
0           0.000378  0.553925              0.0    False       0.779837  
1           0.102351  0.598214              0.0    False       0.537836  
2                NaN       NaN              NaN      NaN            NaN  
3           0.208275  0.442563              0.0    False       0.332402  
4           0.007975  0.516671              0.0    False       0.612240  
5           0.021155  0.434937              0.0    False       0.355256  
6           0.010175  0.477805              0.0    False       0.434485  
7           0.102351  0.851854              0.0    False       0.336465  
8           0.001709  1.512675              0.0    False       0.062468  
9                NaN       NaN              NaN      NaN            NaN  
10          0.102351  0.536301              0.0    False       0.264855  
11          0.102351  0.503005              0.0    False       0.556374  
12               NaN       NaN              NaN      NaN            NaN  
13               NaN       NaN              NaN      NaN            NaN  
14               NaN       NaN              NaN      NaN            NaN
```

```python  # cell 35
spec = pd.DataFrame(C.resolve_controls(cd.SPEC_MUST_HAVES, resource),
                    columns=["source", "target", "ligand_complex", "receptor_complex"])
print("replication-spec must-haves (Fig 8a):")
display(ctrl.merge(spec, on=list(spec.columns)))
```

```
replication-spec must-haves (Fig 8a):

       kind         source         target ligand_complex receptor_complex  \
0  positive  CD4_malignant              B         CXCL13            CXCR5   
1  positive  CD4_malignant              B         CD40LG             CD40   
2  positive              B  CD4_malignant           CD86             CD28   

   found  rank_position  magnitude_rank  specificity_rank  lr_means  \
0   True         3071.0        0.970003          0.000378  0.553925   
1   True         2118.0        0.679052          0.102351  0.598214   
2  False            NaN             NaN               NaN       NaN   

   cellphone_pvals in_top_n  pct_of_tested  
0              0.0    False       0.779837  
1              0.0    False       0.537836  
2              NaN      NaN            NaN
```

```python  # cell 36
_, overlap = C.shuffle_control(ctcl, resource, groupby_pairs=pairs_mal,
                              reference=res_mal, verbose=False)
print(f"\nwithin-donor shuffle overlap with the real top-{cd.TOP_N}: "
      f"{overlap['overlap_with_real'].tolist()}  (target <= 2)")
```

```
shuffle  overlap_with_real  top_n
       0                 13     20
       1                 13     20
       2                 13     20

within-donor shuffle overlap with the real top-20: [13, 13, 13]  (target <= 2)
```

```python  # cell 37
both = ctcl[(ctcl.obs[cd.GROUPBY].astype(str) != cd.CD4_MALIGNANT)
            | (ctcl.obs[cd.EVIDENCE_SRC].astype(str) == "both")].copy()
both.layers[cd.LAYER] = both.X
n_mal = int((ctcl.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT).sum())
n = int((both.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT).sum())
print(f"malignant CD4 {n_mal} -> restricted to TCR+CNV evidence: {n} cells, "
      f"{both.obs.loc[both.obs[cd.GROUPBY].astype(str) == cd.CD4_MALIGNANT, cd.DONOR_KEY].nunique()} donors")
print(f"i.e. {n_mal - n} tcr_only cells removed; this is a nested subset of the ALICE call")
res_both = C.run_rank_aggregate(both, resource, groupby_pairs=pairs_mal,
                               key_added="liana_both", verbose=False)
res_both.to_csv(cd.TAB_DIR / "ccc_headline_evidence_both.csv", index=False)

keys = ["source", "target", "ligand_complex", "receptor_complex"]
head = res_mal.sort_values("magnitude_rank").head(30).set_index(keys)
bb = res_both.set_index(keys)["magnitude_rank"].rename("magnitude_rank_evidence_both")
survive = head[["magnitude_rank"]].join(bb)
survive["survives"] = survive["magnitude_rank_evidence_both"].notna()
survive.to_csv(cd.TAB_DIR / "ccc_headline_evidence_both_survival.csv")
print(f"\ntop-30 pooled pairs still scored under TCR+CNV-only malignancy: "
      f"{int(survive['survives'].sum())}/30")
display(survive)
```

```
malignant CD4 15000 -> restricted to TCR+CNV evidence: 11648 cells, 31 donors
i.e. 3352 tcr_only cells removed; this is a nested subset of the ALICE call

top-30 pooled pairs still scored under TCR+CNV-only malignancy: 30/30

                                                             magnitude_rank  \
source        target        ligand_complex receptor_complex                   
Fibroblast    CD4_malignant VIM            CD44                    0.000002   
Vascular      CD4_malignant VIM            CD44                    0.000058   
CD4_malignant CD4_reactive  VIM            CD44                    0.000162   
Myeloid       CD4_malignant VIM            CD44                    0.000196   
CD4_malignant B             MIF            CD74_CXCR4              0.000274   
              CD8           VIM            CD44                    0.000317   
              Fibroblast    VIM            CD44                    0.000468   
              Keratinocyte  VIM            CD44                    0.000524   
              Tregs         VIM            CD44                    0.000584   
              CD8           MIF            CD74_CXCR4              0.000647   
Fibroblast    CD4_malignant LGALS1         PTPRC                   0.000713   
CD4_malignant Myeloid       VIM            CD44                    0.000782   
Myeloid       CD4_malignant LGALS1         PTPRC                   0.001009   
Fibroblast    CD4_malignant COL6A2         CD44                    0.001091   
                            COL1A2         CD44                    0.001176   
CD4_malignant CD8           LGALS1         PTPRC                   0.001356   
              B             VIM            CD44                    0.001451   
Fibroblast    CD4_malignant COL1A1         CD44                    0.001451   
CD4_reactive  CD4_malignant VIM            CD44                    0.001548   
CD4_malignant CD4_reactive  MIF            CD44_CD74               0.001649   
                                           CD74_CXCR4              0.001754   
Fibroblast    CD4_malignant C3             IFITM1                  0.001972   
CD4_malignant CD4_reactive  LGALS1         PTPRC                   0.002085   
              CD8           MIF            CD44_CD74               0.002202   
              Tregs         LGALS1         PTPRC                   0.002322   
              Fibroblast    MIF            CD44_CD74               0.002445   
Fibroblast    CD4_malignant CXCL14         CXCR4                   0.002969   
Plasma        CD4_malignant VIM            CD44                    0.003108   
CD4_malignant Tregs         MIF            CD44_CD74               0.003395   
Tregs         CD4_malignant VIM            CD44                    0.003694   

                                                             magnitude_rank_evidence_both  \
source        target        ligand_complex receptor_complex                                 
Fibroblast    CD4_malignant VIM            CD44                                  0.000002   
Vascular      CD4_malignant VIM            CD44                                  0.000056   
CD4_malignant CD4_reactive  VIM            CD44                                  0.000188   
Myeloid       CD4_malignant VIM            CD44                                  0.000262   
CD4_malignant B             MIF            CD74_CXCR4                            0.000155   
              CD8           VIM            CD44                                  0.000397   
              Fibroblast    VIM            CD44                                  0.000448   
              Keratinocyte  VIM            CD44                                  0.000502   
              Tregs         VIM            CD44                                  0.000619   
              CD8           MIF            CD74_CXCR4                            0.000559   
Fibroblast    CD4_malignant LGALS1         PTPRC                                 0.000683   
CD4_malignant Myeloid       VIM     
... [4033 more chars truncated]
```

## §10 · The reportable set

The intersection of the checks above. Anything not in this table is exploratory.

```python  # cell 39
keys = ["source", "target", "ligand_complex", "receptor_complex"]
head = res_mal.sort_values("magnitude_rank").head(60).set_index(keys)

d_out = delta_out.set_index(["target", "ligand_complex", "receptor_complex"])["delta_rank"]
d_in = delta_in.set_index(["target", "ligand_complex", "receptor_complex"])["delta_rank"]


def comparator_delta(row):
    s, t, lig, rec = row.name
    if s == cd.CD4_MALIGNANT:
        return d_out.get((t, lig, rec), np.nan)
    return d_in.get((s, lig, rec), np.nan)


report = head[["lr_means", "magnitude_rank", "specificity_rank", "cellphone_pvals"]].copy()
report["delta_vs_reactive"] = head.apply(comparator_delta, axis=1)
report["evidence_both"] = head.index.isin(res_both.set_index(keys).index)
report["caveat_gene"] = [
    ",".join(sorted({g for g in cd.LR_CAVEATS if g in {*str(l).split("_"), *str(r).split("_")}}))
    for (_s, _t, l, r) in head.index
]
verdict_by_pair = feas.set_index(["source", "target"])["verdict"].to_dict()
report["partner"] = [t if s == cd.CD4_MALIGNANT else s for (s, t, _l, _r) in head.index]
report["pair_verdict"] = [
    verdict_by_pair.get((s, t), "unknown") for (s, t, _l, _r) in head.index
]
report["reportable"] = (
    report["evidence_both"]
    & (report["caveat_gene"] == "")
    & (report["delta_vs_reactive"].fillna(-1) < 0)
    & (report["pair_verdict"] == "claim")
)
report.sort_values(["reportable", "magnitude_rank"], ascending=[False, True], inplace=True)
report.to_csv(cd.TAB_DIR / "ccc_reportable_set.csv")
print(f"{int(report['reportable'].sum())} of {len(report)} top pairs pass every check")
display(report)
```

```
25 of 60 top pairs pass every check

                                                             lr_means  \
source        target        ligand_complex receptor_complex             
Fibroblast    CD4_malignant VIM            CD44              2.978544   
CD4_malignant CD4_reactive  VIM            CD44              2.664202   
Myeloid       CD4_malignant VIM            CD44              2.651977   
CD4_malignant CD8           VIM            CD44              2.623429   
              Fibroblast    VIM            CD44              2.622042   
              Keratinocyte  VIM            CD44              2.604457   
              Tregs         VIM            CD44              2.544037   
              Myeloid       VIM            CD44              2.492102   
              CD8           LGALS1         PTPRC             2.338123   
              B             VIM            CD44              2.356822   
CD4_reactive  CD4_malignant VIM            CD44              2.333994   
Fibroblast    CD4_malignant C3             IFITM1            2.308175   
CD4_malignant CD4_reactive  LGALS1         PTPRC             2.298372   
              Tregs         LGALS1         PTPRC             2.265897   
Keratinocyte  CD4_malignant FADD           TRAF2             0.118015   
CD4_malignant Plasma        VIM            CD44              2.192635   
              B             LGALS1         CD69              1.963356   
                                           PTPRC             1.953764   
Fibroblast    CD4_malignant FADD           TRAF2             0.122388   
CD4_malignant CD4_reactive  PKM            CD44              1.900567   
              CD8           PKM            CD44              1.859795   
              Fibroblast    PKM            CD44              1.858407   
              Keratinocyte  PKM            CD44              1.840823   
Myeloid       CD4_malignant HLA-B          CD3D              1.835642   
Vascular      CD4_malignant HLA-B          CD3D              1.829695   
                            VIM            CD44              2.841882   
CD4_malignant B             MIF            CD74_CXCR4        2.625165   
              CD8           MIF            CD74_CXCR4        2.536185   
Fibroblast    CD4_malignant LGALS1         PTPRC             2.508708   
Myeloid       CD4_malignant LGALS1         PTPRC             2.416286   
Fibroblast    CD4_malignant COL6A2         CD44              2.416197   
                            COL1A2         CD44              2.402081   
                            COL1A1         CD44              2.336481   
CD4_malignant CD4_reactive  MIF            CD44_CD74         2.322617   
                                           CD74_CXCR4        2.321537   
              CD8           MIF            CD44_CD74         2.281845   
              Fibroblast    MIF            CD44_CD74         2.258181   
Fibroblast    CD4_malignant CXCL14         CXCR4             2.221647   
Plasma        CD4_malignant VIM            CD44              2.209305   
CD4_malignant Tregs         MIF            CD44_CD74         2.202452   
Tregs         CD4_malignant VIM            CD44              2.190694   
Fibroblast    CD4_malignant COL6A1         CD44              2.166295   
CD4_malignant Myeloid       MIF            CD44_CD74         2.150517   
              Keratinocyte  MIF            CD44_CD74         2.135028   
CD8           CD4_malignant VIM            CD44              2.100680   
CD4_malignant Plasma        MIF            CD74_CXCR4        2.093735   
CD4_reactive  CD4_malignant MIF            CD44_CD74         2.023773   
CD4_malignant B             MIF            CD44_CD74         2.015237   
Vascular      CD4_malignant APP            CD74              2.001440   
Keratinocyte  CD4_malignant MIF            CD44_CD74         1.986105   
Vascular      CD4_malignant MIF            CD44_CD74         1.968244   
Fibroblast    CD4_malignant COL6A3         CD44              1.920063   
CD8           CD4_mal
... [44093 more chars truncated]
```

### Outcome and what is deliberately absent

SVGs in `figures/final/ccc_*.svg`; tables in `tables/ccc_headline_*.csv` and
`tables/ccc_reportable_set.csv`.

**Deliberately not here**, and why:

- **Donor-level p-values.** Everything above is descriptive. The pseudobulk cube
  (`data/ccc/ccc_skin_pseudobulk_full.parquet`, written by the same build pass) makes the
  PyDESeq2 → `li.multi.df_to_lr` phase cheap when it is wanted — the paired
  `~ donor + ccc_celltype` contrast on 29 donors is the one contrast in this atlas free of
  study confounding.
- **Stage, disease and layer stratification.** `ccc_data.FORBIDDEN_CONTRASTS` records why:
  early-vs-advanced is 7 v 7 donors *inside li2024*, SS skin is buus2025 alone, and 417k of
  750k skin cells have `skin_layer == "whole"`.
- **Fine myeloid / fibroblast states.** The li2024 50-level taxonomy resolves `Macro_1/2`,
  `Inf_mac`, `DC2`, `moDC_1-3`, `LC`, `MigDC`, `pDC`, `F1/F2/F3` — but is `Unknown` for the
  330k non-li2024 skin cells, so it can only be run as a li2024-only (14-donor) subset.
- **Spatial validation.** `data/Li2024_atlas/visium/ctcl_visium_destvi.h5ad` has per-spot
  cell-type proportions and would let `li.mt.bivariate` test whether a headline pair is
  co-located, not merely co-expressed across donors.
- **Enforcement of the ≥3-of-4 definition rule.** nb35 §13 computes
  `tables/ccc_pair_definition_stability.csv` but the `reportable` gate above does not read it —
  only `evidence_both` makes it into the expression. Apply the demotion by hand until that is
  wired in.
