# NB28 panel extension — add genes only

## Scope

Add the genes below to the NB28 degradome panel and run Sections A and B on them (and Section C on the appended pairs, §4). **Change nothing else**: same atlas v2 object, same `mal_tcr` malignancy call (no-CDR3 cells excluded), same gates (≥25 cells/arm; ≥4 donors with ≥25 cells), same tests (A: paired Wilcoxon + sign test; B: donor-level Mann-Whitney), same blocks, same external-reference handling, same specificity nulls.

## Rules

1. **Dedupe** against the existing 474-gene panel. Skip genes already present and list them in the report.
2. **Missing genes**: if absent from the atlas matrix or from reynolds2021, keep the row and flag `absent_matrix` / `absent_ext_ref` (as for MMP26/TMEM2). No substitutes.
3. **Multiple testing**: BH-correct the added genes as their **own family** within each section/block. Do not re-correct the original 474 — NB28 q-values must stay reproducible.
4. **Abstentions are expected** for low-copy secreted genes (CD40LG, OSM, IL11, TGFB3, IL17A/F, IL22). Report them as abstained. Do not relax gates.
5. Every added gene is tested in all blocks, as the panel already does. The *compartment* heading below says where the result is interpreted, not where it is restricted.
6. Annotate each gene with `family`, `side` (`CD4-emitted` / `stroma-answered` / `control`) and `partner`.

---

## 1. Malignant clone — interpret in Section A (paired) and Section B `cd4`

**Tier 1 — partner receptor already up in NB28 fibroblasts**

| gene | family | side | partner |
|---|---|---|---|
| CD40LG | juxtacrine | CD4-emitted | CD40 |
| LTA | juxtacrine | CD4-emitted | LTBR |
| LTB | juxtacrine | CD4-emitted | LTBR — on ambient list; trust Section A only |
| OSM | gp130_stat3 | CD4-emitted | OSMR |
| TGFB1 | tgfb_activation | CD4-emitted | TGFBR2 |
| TGFB3 | tgfb_activation | CD4-emitted | TGFBR2 |
| LRRC32 | tgfb_activation | CD4-emitted | — (GARP, latent TGF-β docking) |
| ITGAV | tgfb_activation | CD4-emitted | — (αvβ8 latent TGF-β activation) |
| ITGB8 | tgfb_activation | CD4-emitted | — |
| THBS1 | tgfb_activation | CD4-emitted | — |
| TGM2 | tgfb_activation | CD4-emitted | — |
| PDGFB | caf_expansion | CD4-emitted | PDGFRB |

**Tier 2 — fibroblast-activating ligands not in the panel**

| gene | family | side | partner |
|---|---|---|---|
| SPP1 | ecm_signalling | CD4-emitted | CD44, ITGAV/ITGB1 |
| ENPP2 | lipid_purinergic | CD4-emitted | LPAR1 |
| ENTPD1 | lipid_purinergic | CD4-emitted | ADORA2B |
| NT5E | lipid_purinergic | CD4-emitted | ADORA2B |
| AREG | egfr_ligands | CD4-emitted | EGFR |
| HBEGF | egfr_ligands | CD4-emitted | EGFR |
| IL11 | gp130_stat3 | CD4-emitted | IL11RA |
| VEGFA | angiogenic | CD4-emitted | — |
| CCL2 | chemokine | CD4-emitted | CCR2 |
| CXCL13 | chemokine | CD4-emitted | — |
| WNT5A | wnt_notch | CD4-emitted | ROR2 |
| JAG1 | wnt_notch | CD4-emitted | NOTCH1–3 |
| IL17A | il17_il22 | CD4-emitted | IL17RA, IL17RC |
| IL17F | il17_il22 | CD4-emitted | IL17RA, IL17RC |
| IL22 | il17_il22 | CD4-emitted | IL22RA1 |
| IL26 | il17_il22 | CD4-emitted | — |

**Tier 3 — controls, not mechanism**

| gene(s) | family | purpose |
|---|---|---|
| SLC16A1, SLC16A3 | metabolic_control | decides whether clone BSG is metabolic (MCT chaperone) or EMMPRIN |
| LDHA, ENO1, PKM, SLC2A1, HIF1A, MYC | metabolic_control | tests the proliferation/glycolysis reading of the Section A list |
| GATA3, IRF4, TOX, BATF, RUNX2, STAT3, RELB | regulators | clone state |

## 2. Fibroblasts — interpret in Section B `fibro_low` / `fibro_high`

Receptors for the new ligands, plus the apCAF class-II check.

| gene | family | side | partner |
|---|---|---|---|
| LPAR1 | lipid_purinergic | stroma-answered | ENPP2 |
| ADORA2B | lipid_purinergic | stroma-answered | ENTPD1 / NT5E |
| EGFR | egfr_ligands | stroma-answered | AREG / HBEGF |
| IL11RA | gp130_stat3 | stroma-answered | IL11 |
| IL17RA, IL17RC | il17_il22 | stroma-answered | IL17A / IL17F |
| IL22RA1 | il17_il22 | stroma-answered | IL22 — expected low in fibroblasts |
| ROR2 | wnt_notch | stroma-answered | WNT5A |
| NOTCH1, NOTCH2, NOTCH3 | wnt_notch | stroma-answered | JAG1 |
| HLA-DRA, HLA-DRB1, CIITA | mhc_ii | stroma-answered | read with CD74 + CTSS (already in panel); prefer HLA-DRA — HLA-DRB1 is polymorphic and maps unevenly |

Already in panel, no action: CD40, LTBR, OSMR, TGFBR2, PDGFRB, CD44, ITGB1.

## 3. Myeloid — interpret in Section B `myeloid_low` / `myeloid_high`

| gene | family | side | purpose |
|---|---|---|---|
| CCR2 | chemokine | stroma-answered | CCL2 receptor |
| ADORA2A | lipid_purinergic | stroma-answered | adenosine receptor |

Also pull the myeloid rows for **OSM, SPP1, TGFB1, AREG, VEGFA, CCL2** (added in §1). These are classic myeloid products — the myeloid result is what tells us whether the clone or the myeloid compartment is the dominant source of each signal.

## 4. Section C — append pairs, method unchanged

Append to the curated pair list; same edge construction (sqrt(ligand × receptor) pseudobulk per biopsy), same paired test, same two abundance-matched nulls, BH as a separate family:

`CD40LG→CD40` · `LTA→LTBR` · `LTB→LTBR` · `OSM→OSMR` · `TGFB1→TGFBR2` · `PDGFB→PDGFRB` · `ENPP2→LPAR1` · `NT5E→ADORA2B` · `AREG→EGFR` · `HBEGF→EGFR` · `IL11→IL11RA` · `SPP1→CD44` · `IL17A→IL17RA` · `JAG1→NOTCH3` · `WNT5A→ROR2`

Skip autocrine CD4→CD4 pairs (TIMP1→CD63, PPIA→BSG) — the current design is CD4→stroma.

## 5. Outputs

Write alongside NB28, **do not overwrite** `deg28_*`:

- `deg28x_A_paired_genes.csv`, `deg28x_B_arm_tests.csv`, `deg28x_C_ccc_edges.csv`, `deg28x_coverage.csv`, `deg28x_abstentions.csv` — same schemas as the `deg28_` files, plus `family`, `side`, `partner` columns.
- `deg28x_report.txt`: genes added / already present / `absent_matrix` / `absent_ext_ref`; per gene, tested vs abstained in each section and block.

~59 candidate genes before dedupe.
