# deg42 run log

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

per-cell, descriptive: violin = cells, tick = median, diamond = mean; no cell-level p-values. MF/HC arms are unblocked by study and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title.
```

```
==============================================================================================================
NB28 §4d — CD4 MALIGNANT vs REACTIVE, PAIRED WITHIN SAMPLE
==============================================================================================================
cohort imbalance: 94 MF samples; CD4_malignant per sample median 1,224 / max 17,958; top 10 samples hold 39% of the malignant cells; malignant fraction of CD4 0.16–0.98; li2024 supplies 54% of them
design: replicate = sample; one malignant-minus-reactive delta per biopsy over 25 scores; Wilcoxon signed-rank + sign test, BH over the score set
mal_knn: n_pairs up to 93, 13/25 scores at q<0.05, 9 abstained
mal_tcr: n_pairs up to 33, 7/25 at q<0.05; direction agrees with mal_knn on 12 of 13 scores testable in both
no family mean (below 5 genes): ['caspase', 'glycosidase', 'ha_synthase', 'mast']

donor, study, chemistry, biopsy site and treatment are fixed by construction here -- the two arms are the same tube -- so no sampling imbalance can create a direction. n_pairs is printed beside every p because it is much smaller than the donor count sections 4 and 4b quote. These panels are zero-inflated, so median_delta is often exactly 0 while the signed-rank is decisive; direction is n_pos vs n_neg, and a score with no nonzero paired delta abstains rather than reporting a direction.
```

```
==============================================================================================================
NB28 — CUTANEOUS DEGRADOME BY CELL TYPE, MF + HC SKIN (atlas v2)
==============================================================================================================
scope: ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies
malignancy: headline=mal_knn; HC false-positive floor 27.4%
claimable levels: 18 of 19 (below gate: ['Neutrophils'])
external: reynolds2021 (E-MTAB-8142), 188,698 cells; panel genes absent: ['MMP26', 'TMEM2']
families: 18 (mmp, cathepsin, serine_other, metallo_other, granzyme, inhibitor, hyaluronan, crosslink, glycosidase, collagen_fibrillar, collagen_bmz, collagen_minor, proteoglycan, elastic_fiber, matricellular, adhesion_ligand, matrix_receptor, motility)
cd4: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
myeloid_low: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
myeloid_high: LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | cDC1 [MF 7,500c/84d, HC 172c/20d, external HC 604c/5d] | cDC2 [MF 12,961c/88d, HC 1,171c/22d, external HC 724c/5d] | DC_LAMP3 [MF 21,952c/88d, HC 229c/14d, external HC 2,080c/5d] | pDC [MF 871c/75d, HC 12c/8d, external HC none] | Mono_CD14 [MF 3,589c/82d, HC 7c/7d, external HC 13,890c/5d] | Mono_CD16 [MF 278c/46d, HC 8c/3d, external HC 13,890c/5d] | moDC [MF 6,399c/82d, HC 443c/20d, external HC 12,490c/5d] | Mac_FOLR2 [MF 16,312c/89d, HC 1,272c/21d, external HC 13,890c/5d] | Mac_SPP1_TREM2 [MF 3,470c/81d, HC 368c/16d, external HC 13,890c/5d] | Mac_infl [MF 8,651c/78d, HC 142c/10d, external HC 13,890c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
fibro_low: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
fibro_high: F_papillary [MF 9,475c/80d, HC 3,512c/21d, external HC 17,456c/5d] | F_reticular [MF 18,910c/76d, HC 6,226c/20d, external HC 17,456c/5d] | F_mesenchymal [MF 5,893c/77d, HC 2,366c/21d, external HC 17,456c/5d] | F_myofibroblast [MF 510c/34d, HC 87c/17d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
pooled external comparators: ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']

violins per cell (tick = median, diamond = mean); brackets = donor-level Mann-Whitney on per-donor means, BH per figure, no cell-level p-values. Unblocked by study: MF/HC mix cohorts and the external arm is a different lab, chemistry and sex -- read shifts against the donor counts in each row title. All pairs: tables/deg28_gene_tests/.
```

```
==============================================================================================================
NB28 §9 — MALIGNANT CD4 -> STROMAL FIBROSIS, EIGHT AXES
==============================================================================================================
axes: 8 (th2_fibrotic, gp130_stat3, tgfb_activation, juxtacrine, infl_matrix_suppression, caf_expansion, bsg_resolution, myeloid_innate) over 82 distinct symbols
no axis mean (below 5 genes), read per gene: ['caf_expansion', 'bsg_resolution']
axis genes absent from the external reference: none
cd4_fib: CD4 [malignant 214,460c/81d, reactive 108,258c/81d, external HC 20,617c/5d]
fibro_fib: F_structural [MF 34,788c/87d, HC 12,191c/21d, external HC 17,456c/5d] | F_inflammatory [MF 35,381c/85d, HC 9,856c/21d, external HC 17,456c/5d] | F_apCAF [MF 16,707c/86d, HC 1,093c/21d, external HC 17,456c/5d]
myeloid_fib: Mono_Mac [MF 38,699c/89d, HC 2,240c/21d, external HC 26,380c/5d] | DC [MF 43,284c/90d, HC 1,584c/22d, external HC 3,408c/5d] | LC [MF 3,162c/76d, HC 94c/17d, external HC 12,545c/5d] | Neutrophils [MF 292c/22d, HC 1c/1d, external HC none]
paired within sample (CD4, mal_knn): n_pairs up to 93; CD4-side genes 10/32 at q<0.05, 17 abstained (never detected in either CD4 arm)
direction control: strict clean; soft ['gene_LGALS1', 'gene_LGALS3', 'gene_LGALS9', 'gene_MIF']

the axis MEAN pools both sides and is context only; the CD4-side per-gene rows are the claim. The CD4 side of an axis is evidence only in the cd4_fib rows and the receptor/product side only in fibro_fib / myeloid_fib. Per-gene arm tests: tables/deg28_gene_tests/deg28_{cd4,fibro,myeloid}_fib_*.csv
```

```
==============================================================================================================
NB28 §10 — CURATED LR EDGES, MALIGNANT <-> FIBROBLAST / MYELOID
==============================================================================================================
expression: skin_degradome_pseudobulk_full.parquet (full-gene level x sample count cube from the same build pass), CP10K + log1p per pseudo-sample, size factor over all 42,347 genes; MF only, levels above 25 cells
cohort: 103 samples, 90 donors, 11 studies
panel: ccc_panel.PRIORITY_PAIRS, 15 curated -> 50 concrete edges x 19 stromal levels -> 326 tested edge x level rows after the 5-paired-sample gate
not scored: 2 entries -> tables/deg28_ccc_pairs_dropped.csv (ADAM17->sIL6R (trans-signalling by a shed receptor, not a transcript pair); TCR (a rearranged receptor, not a gene symbol))
score: sqrt(ligand * receptor); a complex takes the min over its subunits
test: the §4d design -- one malignant-minus-reactive delta per biopsy, Wilcoxon signed-rank + sign test, BH; median n_pairs 48; 192 of 326 at q<0.05 (140 up in the malignant arm), 1 abstained

the two directions are NOT symmetric: in stroma_to_cd4 the stromal ligand is identical between the arms (same sample, same level), so that delta is driven entirely by the CD4 receptor and reads as 'does the clone carry more of the receptor', not as a statement about stromal output.
NO specificity rank, NO permutation p, NO null over the rest of the resource -- this is an abundance product over a 15-pair curated panel. A high score means both partners are abundant in the same biopsy, not that the edge is specific to those two levels. NOT comparable to tables/ccc40_* (27_ccc_subtype, LIANA, and a v1-era object).
figures: deg28_ccc_delta_* carries the test; deg28_ccc_chord_*, deg28_ccc_network and deg28_ccc_hierarchy_* are summaries of the same numbers with no test attached.
```

```
==============================================================================================================
NB28 — DEGRADOME, THREE COMPARISONS ON mal_tcr
==============================================================================================================
####################################################################################################
NB28 — THE CUTANEOUS DEGRADOME IN MF SKIN (atlas v2) — RESULTS
####################################################################################################
SCOPE        ['MF', 'HC'] skin | 985,273 cells, 112 donors, 123 samples, 12 studies | panel 474 genes in 23 families
CALL         mal_tcr (ALICE TCR clonality) ONLY. arms = clonal vs non-clonal, MF samples; CD4 without a recovered CDR3 is excluded, not counted as benign.
             mal_knn is QC here and nothing reads it: it calls 27.4% of HEALTHY CD4 malignant. mal_tcr calls 0.0%.
             TCR cohort: MF samples with TCR 38 of 103; donors 38; studies ['chennareddy2025', 'gaydosik2022', 'herrera2021', 'il4ra2026', 'li2024']; clonal cells 68843; non-clonal cells 59220; excluded, no CDR3 195412
UNITS        log1p CP10K, size factor over all 42,347 genes BEFORE the panel subset (gate: 0).
EXTERNAL     reynolds2021 (E-MTAB-8142), 188,698 cells, 5 donors; pooled comparators ['F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'F_structural', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_CD14', 'Mono_CD16']; no external arm: pDC, Neutrophils.
DESIGN       A and C: replicate = SAMPLE, one clonal-minus-non-clonal delta within each biopsy; Wilcoxon signed-rank + sign test. Donor, study, chemistry, site and treatment are fixed by construction.
             B: replicate = DONOR, two-sided Mann-Whitney on per-donor means, >= 4 donors per arm. BH per block (row x gene x arm pair).
CAVEATS      (1) B is NOT blocked by study: MF vs HC mixes cohorts and the external arm cannot be blocked at all -- a significant row means the donor distributions separate.
             (2) These panels are zero-inflated: median_delta is often exactly 0 while the test is decisive. DIRECTION IS pct_up (share of paired samples that moved up), not the median.
             (3) An abstention is a coverage statement, not a null. They are listed at the bottom and never mixed with results.
             (4) C's CD4-side null is drawn from the panel, not the transcriptome, because the TCR-split CD4 rows only exist for the panel. That makes spec_p_cd4_side CONSERVATIVE: it under-calls specificity, it cannot invent it.

====================================================================================================
COMPARISONS RUN
====================================================================================================
§                     comparison                              arms replicates  tested  significant  not tested
A             CD4, within biopsy               clonal - non-clonal       n=33      86           35         388
B                            CD4 clonal / non-clonal / external HC     d=5-38     741          125           0
B      myeloid low res (4 types)             MF / HC / external HC     d=5-84    2672          946        1924
B    myeloid high res (12 types)             MF / HC / external HC     d=5-67    6417         1640        9099
B   fibroblast low res (3 types)             MF / HC / external HC     d=5-59    3535         1611           2
B  fibroblast high res (6 types)             MF / HC / external HC     d=5-51    6609         2649         843
C CD4 <-> fibro/myeloid LR edges               clonal - non-clonal      n<=24     299           51          20

====================================================================================================
A · CD4 CLONAL vs NON-CLONAL, PAIRED WITHIN BIOPSY — 35 genes at q<0.05 (n_pairs <= 33)
====================================================================================================
delta = median over biopsies of (clonal - non-clonal), log1p CP10K. pct_up = share of nonzero paired deltas that went up -- read direction there.
    gene          family   delta  pct_up  n_pairs  n_pos  n_neg      q  sign_p     higher
    PPIA           other  0.4242  1.0000       33     33      0 0.0000  0.0000     clonal
   CXCR4           other -0.8131  0.0645       33      2     29 0.0000  0.0000 non-clonal
     MIF           other  0.3276  0.9091       33     30      3 0.0000  0.0000     clonal
  LGALS3           other  1.1806  0.9167       33     22      2 0.0001  0.0000     clonal
    CD3E           other -0.2326  0.1515       33      5     28 0.0001  0.0001 non-clonal
    IL7R           other -0.8141  0.1786       33      5     23 0.0002  0.0009 non-clonal
    CD63           other  0.5881  0.9091       33     20      2 0.0003  0.0001     clonal
    CSTB       inhibitor  0.2604  0.8214       33     23      5 0.0004  0.0009     clonal
   ANXA2           other  0.4907  0.8000       33     24      6 0.0004  0.0014     clonal
     BSG matrix_receptor  0.4952  0.9048       33     19      2 0.0004  0.0002     clonal
     WAS        motility  0.0000  0.9412       33     16      1 0.0004  0.0003     clonal
   TIMP1       inhibitor  0.3148  0.8947       33     17      2 0.0009  0.0007     clonal
    CTSD       cathepsin  0.0000  1.0000       33     13      0 0.0016  0.0002     clonal
  LGALS1           other  0.9217  0.7667       33     23      7 0.0029  0.0052     clonal
    CD27           other  0.0000  0.9231       33     12      1 0.0029  0.0034     clonal
    CD44 matrix_receptor -0.1838  0.2188       33      7     25 0.0038  0.0021 non-clonal
 TNFRSF4           other  0.0000  0.8125       33     13      3 0.0039  0.0213     clonal
    GNLY           other  0.0000  1.0000       33     11      0 0.0047  0.0010     clonal
     ILK        motility  0.0000  1.0000       33     10      0 0.0088  0.0020     clonal
    GZMB        granzyme  0.0000  1.0000       33      9      0 0.0160  0.0039     clonal
  FERMT3        motility  0.0000  1.0000       33      9      0 0.0160  0.0039     clonal
    TLN1        motility  0.0000  0.7692       33     10      3 0.0181  0.0923     clonal
  S100A4           other  0.2742  0.6364       33     21     12 0.0181  0.1628     clonal
    CD47           other  0.0000  0.9091       33     10      1 0.0245  0.0117     clonal
    CTSA       cathepsin  0.0000  1.0000       33      8      0 0.0249  0.0078     clonal
    CTSH       cathepsin  0.0000  1.0000       33      8      0 0.0249  0.0078     clonal
 KIR3DL2           other  0.0000  1.0000       33      8      0 0.0249  0.0078     clonal
   IL2RG           other  0.1434  0.6970       33     23     10 0.0292  0.0351     clonal
    GZMM        granzyme  0.0000  0.9091       33     10      1 0.0292  0.0117     clonal
SERPINB1       inhibitor  0.0000  1.0000       33      7      0 0.0395  0.0156     clonal
 TNFSF14           other  0.0000  1.0000       33      7      0 0.0395  0.0156     clonal
SERPINB9       inhibitor  0.0000  1.0000       33      7      0 0.0395  0.0156     clonal
   IL21R           other  0.0000  1.0000       33      7      0 0.0395  0.0156     clonal
   PTPRC           other -0.1254  0.3333       33     11     22 0.0395  0.0801 non-clonal
    PPIB           other  0.1233  0.6061       33     20     13 0.0438  0.2962     clonal

  by family: other 20; inhibitor 4; motility 4; cathepsin 3; granzyme 2; matrix_receptor 2
  higher in: clonal 30; non-clonal 5
```

```
==============================================================================================================
NB28 — DEGRADOME, THREE COMPARISONS ON mal_tcr
==============================================================================================================
{'§': 'A', 'comparison': 'CD4, within biopsy', 'arms': 'clonal - non-clonal', 'replicates': 'n=33', 'tested': 86, 'significant': 35, 'not tested': 388}
{'§': 'B', 'comparison': 'CD4', 'arms': 'clonal / non-clonal / external HC', 'replicates': 'd=5-38', 'tested': 741, 'significant': 125, 'not tested': 0}
{'§': 'B', 'comparison': 'myeloid low res (4 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-84', 'tested': 2672, 'significant': 946, 'not tested': 1924}
{'§': 'B', 'comparison': 'myeloid high res (12 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-67', 'tested': 6417, 'significant': 1640, 'not tested': 9099}
{'§': 'B', 'comparison': 'fibroblast low res (3 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-59', 'tested': 3535, 'significant': 1611, 'not tested': 2}
{'§': 'B', 'comparison': 'fibroblast high res (6 types)', 'arms': 'MF / HC / external HC', 'replicates': 'd=5-51', 'tested': 6609, 'significant': 2649, 'not tested': 843}
{'§': 'C', 'comparison': 'CD4 <-> fibro/myeloid LR edges', 'arms': 'clonal - non-clonal', 'replicates': 'n<=24', 'tested': 299, 'significant': 51, 'not tested': 20}
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
```
