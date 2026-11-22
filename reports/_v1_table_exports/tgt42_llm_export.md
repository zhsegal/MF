> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 42 — Surface-target screen, skin

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**, 539,916 T/NK cells from
skin_T_tcr_annotated_v4.h5ad (42,347 genes in the object; the screen tests
the 2,355-gene surface panel union surfaceome selection
plus lineage and trafficking controls).

Cohort: 146 donor keys reduced to 109 independent keys
(107 people) by dropping non-CD4 clones,
diagnoses outside the MF/SS spectrum, and repeat biopsies of one patient. Diagnosis from the
curated `clin_dx`. 75 donors carry both a malignant and a reactive CD4 arm and
enter the paired test.

Arms: malignancy call `mal_cluster`; CD4 cells are split into CD4_malignant,
CD4_reactive, CD4_unassessed (never used as a comparator) and CD4_HC (healthy donors), beside
CD4_Treg and CD8.

Headline test: per gene, the within-donor difference of mean log1p CP10K between the malignant
and reactive arms, paired Wilcoxon over donors, BH-adjusted (`q`). `log2fc_paired` is the median
per-donor log2 ratio; `frac_donors_up` the share of donors with a positive difference.
`z_blocked` is the study-blocked van Elteren statistic, which also uses donors with one arm.

Known answers, for calibration: CCR4 (mogamulizumab) and TNFRSF8/CD30 (brentuximab) are approved
in CTCL, KIR3DL2 is lacutamab's target, IL2RA is registrational. CD7 and DPP4/CD26 loss is the
textbook CTCL immunophenotype and must come out negative. The previous version of this screen
failed that check (worst control 1,968 of 2,115), which is what prompted the rebuild.

## §0b Cohort

**What:** which donor keys are independent MF/SS patients with a CD4 clone, and why each dropped key was dropped.

**How:** curated clin_* columns only (docs/CLINICAL_METADATA.md); no expression used.

**Full table on disk:** `tables/tgt42_skin_cohort.csv`

```csv
donor,study,dx,dx2,lesion,lesion_class,clin_stage,clin_stage_class,sex,age,is_cultured,n_cells,burden
B5__PT09,buus2025,SS,SS,erythroderma,lesion,,,F,71,False,18455,0.664
B5__PT22,buus2025,SS,SS,erythroderma,lesion,,,M,70,False,9892,
B5__PT23,buus2025,SS,SS,erythroderma,lesion,,,F,85,False,8229,0.836
D10__JONAK1,jonak2021,MF,MF,plaque,lesion,IB,early,F,33,False,2799,0.153
D11__P1,brentuximab2026,MF,MF,plaque,lesion,IIIA,advanced,F,52,False,898,0.883
D11__P2,brentuximab2026,MF,MF,tumor,lesion,IIIA,advanced,F,48,False,11850,0.899
D11__P3,brentuximab2026,MF,MF,tumor,lesion,IIB,advanced,F,55,False,10804,0.827
D11__P4,brentuximab2026,MF,MF,plaque,lesion,IIIA,advanced,F,45,False,897,0.254
D11__P5,brentuximab2026,MF,MF,tumor,lesion,IVA2,advanced,F,61,False,9000,0.827
D11__P6,brentuximab2026,MF,MF,plaque,lesion,IVA2,advanced,M,64,False,7515,0.769
D13__MF17,gaydosik2022,MF,MF,tumor,lesion,IIB,advanced,F,76,False,3991,0.985
D13__MF18,gaydosik2022,MF,MF,tumor,lesion,IIB,advanced,M,87,False,2677,0.469
D13__MF19,gaydosik2022,MF/SS,unresolved,tumor,lesion,IIIB,advanced,M,62,False,528,0.518
D13__MF21,gaydosik2022,MF,MF,tumor,lesion,IIB,advanced,M,61,False,966,0.177
D13__MF25,gaydosik2022,MF/SS,unresolved,erythroderma,lesion,IVA1,advanced,M,68,False,430,0.354
D13__SC327_HTO1,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,28,
D13__SC327_HTO2,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,225,
D13__SC374_HTO10,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,935,
D13__SC374_HTO9,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,230,
D14__SC320,gaydosik2023,HC,HC,healthy_skin,healthy,,,F,,False,140,
D14__SC348,gaydosik2023,HC,HC,healthy_skin,healthy,,,,,False,494,
D14__SC353,gaydosik2023,HC,HC,healthy_skin,healthy,,,F,,False,253,
D14__SC354,gaydosik2023,HC,HC,healthy_skin,healthy,,,F,,False,111,
D14__SC405,gaydosik2023,HC,HC,healthy_skin,healthy,,,M,,False,149,
D15__MF22,il4ra2026,MF,MF,tumor,lesion,IIB,advanced,M,80,True,31,
D15__MF24,il4ra2026,MF,MF,tumor,lesion,IIB,advanced,M,42,False,3635,0.247
D15__MF26,il4ra2026,MF,MF,erythroderma,lesion,IVB,advanced,M,30,False,3599,0.92
D15__MF28,il4ra2026,MF,MF,tumor,lesion,IIB,advanced,F,74,True,1195,0.151
D15__MF30,il4ra2026,MF,MF,erythroderma,lesion,IIIB,advanced,M,80,True,1047,0.584
D15__MF31,il4ra2026,SS,SS,erythroderma,lesion,IVA1,advanced,M,89,True,50,
D15__MF34,il4ra2026,MF,MF,tumor,lesion,IIB,advanced,M,76,True,1113,0.264
D15__MF35,il4ra2026,MF,MF,mixed,lesion,IIB,advanced,F,66,True,4182,0.786
D15__MF36,il4ra2026,MF,MF,mixed,lesion,IIB,advanced,M,48,False,4311,0.221
D15__MF40,il4ra2026,MF,MF,erythroderma,lesion,IIIB,advanced,M,75,True,818,
D15__MF48,il4ra2026,MF,MF,papule,lesion,IIB,advanced,M,71,False,4681,0.837
D15__NS13,il4ra2026,HC,HC,healthy_skin,healthy,,,F,,False,390,
D15__NS14,il4ra2026,HC,HC,healthy_skin,healthy,,,F,,False,554,
D15__NS16,il4ra2026,HC,HC,healthy_skin,healthy,,,F,,False,91,
D15__NS17,il4ra2026,HC,HC,healthy_skin,healthy,,,F,,False,352,
D1__P112,chennareddy2025,HC,HC,healthy_skin,healthy,,,F,51,False,269,
D1__P115,chennareddy2025,HC,HC,healthy_skin,healthy,,,M,48,False,605,
D1__P116,chennareddy2025,HC,HC,healthy_skin,healthy,,,F,56,False,485,
D1__P121,chennareddy2025,HC,HC,healthy_skin,healthy,,,F,44,False,83,
D1__P171,chennareddy2025,MF,MF,tumor,lesion,IVA1,advanced,M,84,False,6288,0.932
D1__P204,chennareddy2025,MF,MF,tumor,lesion,IIB,advanced,F,71,False,11668,0.375
D1__P303,chennareddy2025,MF,MF,tumor,lesion,IVB,advanced,F,65,False,1524,0.899
D1__P312,chennareddy2025,MF,MF,plaque,lesion,IIB,advanced,M,55,False,1982,0.566
D1__PGS,chennareddy2025,MF,MF,erythroderma,lesion,IVA1,advanced,M,75,False,3193,0.817
D3__P105,brunner2024,MF,MF,erythroderma,lesion,IVA1,advanced,M,84,False,715,0.185
D3__P192,brunner2024,SS,SS,erythroderma,lesion,IVA1,advanced,M,67,False,4227,0.813
D3__P196,brunner2024,MF,MF,erythroderma,lesion,IVA1,advanced,F,75,False,797,0.791
D3__P87,brunner2024,SS,SS,erythroderma,lesion,IVA1,advanced,M,86,False,351,
D3__P94,brunner2024,MF,MF,erythroderma,lesion,IVA1,advanced,M,71,False,3042,
D6__MF2,gaydosik2019,SS,SS,erythroderma,lesion,IVA,advanced,F,68,False,1600,0.489
D6__NOR1,gaydosik2019,HC,HC,healthy_skin,healthy,,,M,64,False,62,
D6__NOR2,gaydosik2019,HC,HC,healthy_skin,healthy,,,F,48,False,66,
D6__NOR3,gaydosik2019,HC,HC,healthy_skin,healthy,,,M,54,False,97,
D6__NOR4,gaydosik2019,HC,HC,healthy_skin,healthy,,,M,61,False,73,
D6__SC157,gaydosik2019,MF,MF,tumor,lesion,IIB,advanced,F,83,False,1730,0.572
D6__SC158,gaydosik2019,MF,MF,plaque,lesion,IIB,advanced,F,46,False,461,0.756
D7__MF309,rindler2021_mc,MF,MF,patch,lesion,IVA1,advanced,M,76,False,8778,0.755
D7__MF311,rindler2021_mc,MF,MF,patch,lesion,IVA1,advanced,F,74,False,9494,0.748
D7__P73,rindler2021_mc,MF,MF,plaque,lesion,IIB,advanced,M,82,False,769,0.327
D7__P84,rindler2021_mc,MF,MF,plaque,lesion,IIB,advanced,M,75,False,6959,0.327
D8__P138,alkon2024,MF,MF,patch,lesion,IA,early,M,47,False,1489,0.505
D8__P150,alkon2024,MF,MF,patch,lesion,IA,early,M,56,False,4061,0.724
D8__P163,alkon2024,MF,MF,patch,lesion,IB,early,F,83,False,1482,0.679
D8__P65,alkon2024,MF,MF,patch,lesion,IB,early,M,53,False,1386,0.496
D8__P90,alkon2024,MF,MF,patch,lesion,IB,early,M,75,False,393,0.227
H__HC1,herrera2021,HC,HC,healthy_skin,healthy,,,M,41,False,25,
H__MFIV1,herrera2021,MF,MF,plaque,lesion,IVB,advanced,F,39,False,98,0.62
H__SS1,herrera2021,SS,SS,patch,lesion,IVB,advanced,F,49,False,137,0.33
H__SS3,herrera2021,SS,SS,plaque,lesion,IVA1,advanced,F,71,False,85,
H__SS4,herrera2021,SS,SS,erythroderma,lesion,IVA2,advanced,M,56,False,288,0.816
Li2024_atlas__CTCL1,li2024,MF,MF,mixed,lesion,IIB,advanced,F,38,False,9232,0.956
Li2024_atlas__CTCL10,li2024,MF,MF,skin_unspecified,unspecified,IB,early,F,56,False,1429,0.88
Li2024_atlas__CTCL11,li2024,MF,MF,skin_unspecified,unspecified,IB,early,F,58,False,102,
Li2024_atlas__CTCL12,li2024,MF,MF,skin_unspecified,unspecified,IV,advanced,M,60,False,3491,0.894
Li2024_atlas__CTCL13,li2024,MF,MF,skin_unspecified,unspecified,IIB,advanced,M,68,False,11240,0.674
Li2024_atlas__CTCL14,li2024,MF,MF,skin_unspecified,unspecified,IB,early,F,68,False,232,0.288
Li2024_atlas__CTCL15,li2024,MF,MF,skin_unspecified,unspecified,IA,early,M,90,False,103,
Li2024_atlas__CTCL16,li2024,MF,MF,skin_unspecified,unspecified,IA,early,F,37,False,1814,0.779
Li2024_atlas__CTCL17,li2024,MF,MF,skin_unspecified,unspecified,IIB,advanced,M,76,False,321,0.712
Li2024_atlas__CTCL18,li2024,MF,MF,skin_unspecified,unspecified,IV,advanced,M,68,False,7545,0.982
Li2024_atlas__CTCL2,li2024,MF,MF,mixed,lesion,IA,early,M,67,False,5355,0.382
Li2024_atlas__CTCL3,li2024,MF,MF,mixed,lesion,IB,early,F,74,False,13308,0.832
Li2024_atlas__CTCL4,li2024,MF,MF,mixed,lesion,IB,early,M,57,False,12894,0.136
Li2024_atlas__CTCL5,li2024,MF,MF,mixed,lesion,IIB,advanced,M,56,False,14809,0.649
Li2024_atlas__CTCL6,li2024,MF,MF,mixed,lesion,IA,early,M,67,False,17933,0.277
Li2024_atlas__CTCL7,li2024,MF,MF,mixed,lesion,IA,early,F,59,False,56311,0.0869
Li2024_atlas__CTCL8,li2024,MF,MF,mixed,lesion,IB,early,M,61,False,12191,0.751
Li2024_atlas__CTCL9,li2024,MF,MF,skin_unspecified,unspecified,IA,early,F,46,False,2585,0.954
Li2024_atlas__MF15,li2024,MF,MF,plaque,lesion,IB,early,M,35,False,1317,
Li2024_atlas__MF17,li2024,MF,MF,tumor,lesion,IIB,advanced,M,36,False,5175,0.00837
Li2024_atlas__MF18,li2024,MF,MF,tumor,lesion,IVA2,advanced,M,70,False,836,
Li2024_atlas__MF21,li2024,MF,MF,tumor,lesion,IIB,advanced,M,79,False,3002,0.713
Li2024_atlas__MF22,li2024,MF,MF,plaque,lesion,IB,early,M,71,False,1172,0.343
Li2024_atlas__MF26,li2024,MF,MF,plaque,lesion,IIA,early,M,15,False,2084,
Li2024_atlas__MF27,li2024,MF,MF,tumor,lesion,IIB,advanced,F,57,False,1746,0.129
Li2024_atlas__MF28,li2024,MF,MF,tumor,lesion,IIIB,advanced,M,60,False,15030,0.845
Li2024_atlas__MF30,li2024,MF,MF,tumor,lesion,IIB,advanced,F,48,False,5017,0.549
Li2024_atlas__PT11,li2024,MF,MF,patch,lesion,IVA2,advanced,M,50,False,2420,0.874
Li2024_atlas__PT35,li2024,MF,MF,mixed,lesion,IIB,advanced,M,61,False,5080,0.919
Li2024_atlas__PT47,li2024,MF,MF,mixed,lesion,IIB,advanced,M,60,False,571,0.65
Li2024_atlas__PT50,li2024,MF,MF,mixed,lesion,IB,early,M,73,False,3833,0.441
Li2024_atlas__PT52,li2024,MF,MF,mixed,lesion,IIB,advanced,F,55,False,266,0.643
Li2024_atlas__PT53,li2024,MF,MF,mixed,lesion,IIB,advanced,M,58,False,2555,0.658
Li2024_atlas__PT55,li2024,MF,MF,mixed,lesion,IIB,advanced,M,71,False,2470,0.968
Li2024_atlas__PT56,li2024,MF,MF,mixed,lesion,IIB,advanced,M,65,False,5592,0.343
```

## §0b Dropped keys

**What:** the keys removed from the cohort, with the reason.

**How:** same source; the three rules are non-CD4 clone, off-spectrum diagnosis, repeat biopsy of a patient already in the cohort.

```csv
donor,study,dx,phenotype,lesion,n_cells,drop_reason
D1__P124,chennareddy2025,MF,gd,plaque,3158,malignant clone is not CD4 (gd)
D1__P127,chennareddy2025,MF,gd,plaque,1772,malignant clone is not CD4 (gd)
D1__P195,chennareddy2025,pcAECTCL,CD8,plaque,11243,malignant clone is not CD4 (CD8)
D1__P202,chennareddy2025,MF,gd,plaque,8040,malignant clone is not CD4 (gd)
D1__P218,chennareddy2025,pcAECTCL,CD8,plaque,2547,malignant clone is not CD4 (CD8)
D1__P230,chennareddy2025,pcAECTCL,CD8,tumor,694,malignant clone is not CD4 (CD8)
D1__P311,chennareddy2025,MF,CD4,plaque,5258,same person as D7__MF311
D1__P318_thick,chennareddy2025,MF,gd,plaque,3759,malignant clone is not CD4 (gd)
D1__P318_thin,chennareddy2025,MF,gd,patch,3683,malignant clone is not CD4 (gd)
D1__P71,chennareddy2025,MF,gd,tumor,548,malignant clone is not CD4 (gd)
D1__P76,chennareddy2025,MF,CD4,tumor,7018,same person as D7__MF309
D1__P95,chennareddy2025,pcAECTCL,CD8,tumor,5843,malignant clone is not CD4 (CD8)
D1__P97,chennareddy2025,MF,gd,plaque,4019,malignant clone is not CD4 (gd)
D6__MF5,gaydosik2019,CD8_aggr_CTCL,CD8,tumor,5350,malignant clone is not CD4 (CD8)
D6__SC205,gaydosik2019,MF,DN,tumor,1486,malignant clone is not CD4 (DN)
D7__MF312,rindler2021_mc,MF,CD4,patch,499,same person as D1__P312
D7__P65,rindler2021_mc,MF,CD4,nonlesional,1927,same person as D8__P65
D7__P90,rindler2021_mc,MF,CD4,nonlesional,129,same person as D8__P90
D8__P102,alkon2024,SPP,not_applicable,patch,1384,diagnosis outside the MF/SS spectrum (SPP)
D8__P107,alkon2024,MF,DN,patch,2020,malignant clone is not CD4 (DN)
D8__P143,alkon2024,LPP,not_applicable,patch,859,diagnosis outside the MF/SS spectrum (LPP)
D8__P144,alkon2024,SPP,not_applicable,patch,106,diagnosis outside the MF/SS spectrum (SPP)
D8__P149,alkon2024,SPP,not_applicable,patch,654,diagnosis outside the MF/SS spectrum (SPP)
D8__P194,alkon2024,LPP,not_applicable,patch,2166,diagnosis outside the MF/SS spectrum (LPP)
D8__P78,alkon2024,SPP,not_applicable,patch,417,diagnosis outside the MF/SS spectrum (SPP)
D8__P79,alkon2024,SPP,not_applicable,patch,1645,diagnosis outside the MF/SS spectrum (SPP)
D8__P88,alkon2024,SPP,not_applicable,patch,865,diagnosis outside the MF/SS spectrum (SPP)
D9__P101,lyp2026,LyP,CD4,LyP_lesion,3509,diagnosis outside the MF/SS spectrum (LyP)
D9__P118,lyp2026,LyP,CD4,LyP_lesion,892,diagnosis outside the MF/SS spectrum (LyP)
D9__P193,lyp2026,LyP,gd,LyP_lesion,11896,diagnosis outside the MF/SS spectrum (LyP)
D9__P213,lyp2026,LyP,CD8,LyP_lesion,7308,diagnosis outside the MF/SS spectrum (LyP)
D9__P217,lyp2026,LyP,CD4,LyP_lesion,625,diagnosis outside the MF/SS spectrum (LyP)
D9__P220,lyp2026,LyP,CD8,LyP_lesion,5259,diagnosis outside the MF/SS spectrum (LyP)
D9__P72,lyp2026,MF,CD4,erythroderma,2671,same person as D7__MF311
D9__PEN,lyp2026,LyP,CD8,LyP_lesion,4739,diagnosis outside the MF/SS spectrum (LyP)
H__SS2,herrera2021,SS,CD4,erythroderma,191,same person as D13__MF19
Li2024_atlas__MF14,li2024,MF,CD8,plaque,4369,malignant clone is not CD4 (CD8)
```

## §1 Arms

**What:** donors, cells and studies per arm after the cohort fix.

**How:** groups of >= 20 cells.

```csv
arm,donors,cells,studies
CD4_malignant,78,187748,12
CD4_reactive,82,147233,12
CD8,86,45562,13
CD4_Treg,77,35560,13
CD4_HC,21,4724,5
CD4_unassessed,2,71,1
```

## §1 Contamination load per arm

**What:** how dirty each arm is: share of lineage-positive cells, library size, doublet score and the tissue module scores.

**How:** median over the arm's donors. A malignant arm dirtier than its reactive arm is the precondition for the contamination story in §2.

```csv
arm,frac_lineage_pos,median_total_counts,median_doublet_score,score_keratinocyte,score_fibroblast,score_myeloid,score_endothelial,score_melanocyte,score_mast
CD4_HC,0.367,2.37e+03,0.0183,0.187,0.0403,0.0426,0.0117,0.00113,0.000992
CD4_Treg,0.309,2.55e+03,0.0153,0.0859,0.0119,0.0198,0.0055,0.0014,0
CD4_malignant,0.427,3.33e+03,0.0205,0.116,0.0255,0.0372,0.00999,0.00198,0.000677
CD4_reactive,0.332,2.04e+03,0.0138,0.0962,0.0152,0.0624,0.0163,0.00168,0.000738
CD4_unassessed,0.702,1.27e+04,0.0311,0.214,0.0224,0.0662,0.0264,0.00827,0
CD8,0.305,2.11e+03,0.0146,0.112,0.0157,0.039,0.0349,0.000519,0
```

## §2a Contamination null

**What:** what pure contamination scores here: the six statistics computed on transcripts a CD4 T cell cannot make (keratins, collagens, VWF, PMEL, tryptases, LYZ, HBB).

**How:** same paired test and per-cell couplings as every other gene. Thresholds for the artifact flags are read off the p90 row of this band, not chosen.

```csv
index,gene,log2fc_paired,frac_donors_up,median_pct_expr,retained_trim,retained_clean,retained_ds,top1_share_excess,r_keratinocyte_excess,r_fibroblast_excess,r_myeloid_excess,or_topdecile_excess,rho_burden
548,SFN,0.0455,0.56,0.0634,0,0.546,0,0.0171,0.334,0.00672,0.0114,-0.466,0.296
575,CD68,0.0346,0.533,0.0208,0,0,-1,-0.0498,0.00751,0.00423,0.342,-0.118,-0.203
599,KRT10,0.0261,0.533,0.408,0.646,-1,-1,0,0.719,0.00206,0.00563,-0.354,-0.0942
616,EGFL7,0.0216,0.56,0.00455,0,0,0,0,-0.0043,0.0157,0.00172,0.144,0.116
630,COL1A1,0.0153,0.533,0.0357,0,0,0,0.00292,0.0079,0.59,0.0132,-0.552,-0.0356
668,KRT17,0.0066,0.533,0.0156,0,0,0,-0.0848,0.187,0.000213,0.00765,-0.322,0.0918
673,C1QB,0.00442,0.507,0.00652,0,0,0,0,0.00866,0.00788,0.2,-0.385,0.0327
876,CD14,0,0.453,0.00241,,,,0,0.00158,0.0017,0.153,-0.0764,-0.0252
989,SFRP2,0,0.347,0.00108,,,,0,0.0134,0.277,0.00933,-0.105,0.301
1078,VWF,0,0.427,0.000995,,,,0,-0.000972,-0.00135,0.00256,-0.191,0.0345
1092,TYRP1,0,0.293,0.00117,,,,0,0.0135,0.00155,0.000452,-0.33,-0.000589
1266,PMEL,0,0.493,0.00269,,,,0,0.000894,-0.005,-0.00126,-0.197,0.0547
1332,HBB,0,0.333,0.00597,,,,-0.0503,-0.00424,-0.00888,0.00823,-0.598,0.0561
1333,HBA2,0,0.333,0.00127,,,,0,-0.0115,-0.01,-0.00536,-0.345,0.0741
1405,CPA3,0,0.267,0,,,,0,-0.00201,0.000994,0.0177,-0.163,0.194
1459,COL1A2,0,0.373,0.00469,,,,-0.35,0.0081,0.343,0.0083,-0.299,0.0758
1597,DCT,0,0.387,0.000267,,,,0,0.0155,0.00132,-0.00614,-0.353,0.0499
1644,LUM,0,0.307,0.00118,,,,0,0.00588,0.21,0.00646,-0.446,0.0243
1682,CDH5,0,0.2,0,,,,0,0.00665,-0.000631,-0.000793,-0.0607,0.0328
1710,MS4A2,0,0.253,0,,,,0,-0.00255,0.0194,0.0124,-0.0648,0.301
1729,CD163,0,0.333,0.000373,,,,0,0.00298,-0.0013,0.118,-0.18,0.108
1773,MLANA,0,0.293,0.000322,,,,0,0.0118,0.00215,-0.00112,-0.156,-0.00165
1780,TPSAB1,0,0.227,0,,,,0,0.00533,0.0042,0.0134,-0.331,-0.0537
1781,TPSB2,0,0.187,0,,,,0,0.00406,-0.00151,0.0103,-0.0258,0.122
1801,CLDN5,0,0.493,0.0037,,,,0,-0.0133,0.00416,-0.0128,-0.276,-0.0192
1816,KRTDAP,0,0.4,0.0148,,,,-0.27,0.211,-0.00326,1.11e-05,-0.487,-0.0062
1927,JCHAIN,0,0.347,0.000824,,,,0,0.000568,0.00847,0.0169,-0.246,0.101
1956,COL3A1,-0.000764,0.307,0.00459,,,,-0.333,0.0222,0.368,0.0132,-0.36,0.245
1984,KRT6A,-0.0112,0.32,0.0109,-0,-0,-0,-0.415,0.273,-0.00314,0.0168,-0.412,0.0494
1998,C1QA,-0.0161,0.413,0.00584,-0,-0,-0,0,0.0153,0.00832,0.217,-0.428,-0.0324
2015,DCN,-0.0264,0.307,0.00969,-0,-0,0.741,-0.45,0.0173,0.42,0.0196,-0.359,-0.0411
2053,KRT5,-0.052,0.36,0.0367,-0,-0,0.795,-0.076,0.308,0.00858,0.0088,-0.586,-0.0415
2082,LYZ,-0.0733,0.387,0.0171,-0,-0,1.52,0.0151,0.0144,0.00669,0.325,-0.654,-0.231
2099,CD79A,-0.0873,0.347,0.00509,-0,0.0343,-0,0,-0.00735,-0.00368,0.0183,-0.111,-0.0753
2100,DMKN,-0.0882,0.293,0.0148,-0,-0,1.25,-0.311,0.225,-0.00481,0.0202,-0.294,0.0912
2112,KRT14,-0.0946,0.36,0.07,-0,-0,0.419,0.0363,0.4,-0.000283,0.0194,-0.686,-0.169
2118,MZB1,-0.106,0.427,0.00798,-0,-0,1.12,-0.125,0.00544,-0.0038,0.000728,0.3,-0.179
2161,KRT1,-0.152,0.28,0.0261,0.0891,-0,0.743,-0.00658,0.313,-0.0052,0.00507,-0.695,0.178
2176,COL6A2,-0.17,0.373,0.0262,-0,0.646,1.36,0.0917,0.00715,0.687,0.0345,-0.121,0.00336
2195,MS4A1,-0.203,0.187,0.00187,-0,-0,0.497,0,0.00613,-0.000907,0.00754,-0.0654,0.039
```

## §2a Calibration verdict

**What:** whether the contamination mechanism is operating in this cohort at all.

**How:** the null is calibrated only if the lineage controls are themselves up in the malignant arm (median log2FC > 0.1 and median frac_donors_up > 0.5). If not, no gene is excluded on contamination grounds.

```csv
key,value
null_calibrated,False
verdict,"40 lineage controls: median log2FC +0.00, median frac_donors_up 0.35 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first."
```

## §2b Artifact verdict, per gene

**What:** the six tests and the resulting flag for every gene up at q < 0.05.

**How:** fails_trim / fails_clean / fails_ds are the share of the effect retained after trimming the top 1% of cells, dropping lineage-positive and high-doublet cells, and depth-matching; fails_concentration / fails_lineage / fails_umi are prevalence-matched residuals against the band; fails_burden is the Spearman of the per-donor effect against that donor's malignant:reactive ratio. Two failures = artifact, one = suspect.

**Full table on disk:** `tables/tgt42_artifact_evidence.csv`

```csv
index,gene,log2fc_paired,frac_donors_up,median_pct_expr,retained_trim,retained_wins,retained_clean,retained_ds,top1_share_excess,r_keratinocyte_excess,r_fibroblast_excess,r_myeloid_excess,r_umi_excess,or_topdecile_excess,rho_burden,top_cluster_share,n_fails,fail_reasons
0,CD70,1.1,0.92,0.21,1.1,1.85,0.899,0.693,0.0578,0.00769,0.0228,0.00413,0.0502,0.444,-0.0992,0.01,2,- (null not calibrated; no exclusion applied)
1,CD9,1.08,0.733,0.25,1.13,1.26,0.987,0.906,0.0757,0.0274,0.0231,0.0137,-0.00481,-0.00161,-0.128,0.0124,2,- (null not calibrated; no exclusion applied)
2,CPM,1.02,0.747,0.137,1.11,1.41,0.869,0.792,0.0485,-0.00292,0.00657,-0.00607,-0.00352,0.191,-0.081,0.0113,2,- (null not calibrated; no exclusion applied)
3,MRC2,0.892,0.88,0.0805,1.11,0,1.08,0.812,-0.00191,-0.00593,0.0127,0.00761,0.0149,0.297,0.0363,0.0171,1,- (null not calibrated; no exclusion applied)
4,SLC29A1,0.841,0.933,0.0825,1.04,0,0.807,0.673,0.0771,-0.0128,0.00409,0.00234,0.121,1.1,-0.0381,0.0131,2,- (null not calibrated; no exclusion applied)
5,COL6A3,0.811,0.88,0.113,1.03,0.933,1.15,0.934,0.0229,-0.0139,0.0177,0.00529,-0.0159,0.132,0.0646,0.0208,2,- (null not calibrated; no exclusion applied)
6,CD151,0.794,0.76,0.196,1.15,1.85,0.943,0.791,0.0108,0.0104,0.0115,0.0241,0.0769,0.526,0.0725,0.00822,2,- (null not calibrated; no exclusion applied)
7,ADGRE1,0.792,0.707,0.0487,0.881,0,0.734,0.81,0.00466,0.00785,-0.00575,-0.0098,0.0454,0.207,0.0172,0.0176,2,- (null not calibrated; no exclusion applied)
8,GRN,0.778,0.907,0.211,1.08,1.58,0.886,0.602,0.00173,0.017,0.00404,0.0282,0.0785,0.529,0.0709,0.00836,1,- (null not calibrated; no exclusion applied)
9,NRN1,0.776,0.773,0.0461,0.894,0,0.817,0.873,0.0515,-0.000638,0.0143,-0.00257,0.0343,0.215,0.294,0.0322,2,- (null not calibrated; no exclusion applied)
10,PLD1,0.736,0.747,0.0656,1.15,0,0.845,0.938,0.00852,-0.00529,0.00127,-0.0115,-0.0301,-0.324,0.072,0.02,1,- (null not calibrated; no exclusion applied)
11,TFRC,0.731,0.893,0.296,1.07,1.62,1.06,0.706,0.0211,-0.0289,-0.000481,-0.0141,-0.0662,-0.306,0.0715,0.00919,1,- (null not calibrated; no exclusion applied)
12,SLC39A6,0.707,0.947,0.183,1.15,2,1.08,0.599,-0.00269,-0.0084,-0.00595,-0.000871,0.0523,0.225,-0.0683,0.0114,1,- (null not calibrated; no exclusion applied)
13,SLC4A2,0.698,0.933,0.13,1.09,1.42,1.1,0.819,0.000902,-0.00624,0.00457,-0.00121,0.0364,0.203,-0.000114,0.0111,1,- (null not calibrated; no exclusion applied)
14,TNFRSF25,0.698,0.747,0.339,1.05,1.16,0.855,0.77,-0.00271,-0.0113,-0.00213,-0.0285,-0.0771,-0.347,0.251,0.00926,0,- (null not calibrated; no exclusion applied)
15,SLC1A4,0.69,0.747,0.144,1.09,1.27,0.66,0.534,0.0348,0.000619,0.00748,-0.00382,0.0777,0.579,-0.179,0.0139,2,- (null not calibrated; no exclusion applied)
16,LGALS3,0.663,0.587,0.427,1.07,0.994,0.861,0.917,0.0356,0.0266,-0.00374,0.0148,0.00105,-0.0681,-0.0744,0.00696,1,- (null not calibrated; no exclusion applied)
17,PDIA4,0.637,0.88,0.255,1.11,1.28,0.901,0.595,0.0217,-0.00999,0.0104,-0.00286,0.0458,0.135,-0.0762,0.013,2,- (null not calibrated; no exclusion applied)
18,FUT8,0.635,0.867,0.184,1.1,1.4,1.03,0.692,0.0116,-0.0104,0.00588,-0.00783,-0.0265,0.0656,-0.0842,0.0111,2,- (null not calibrated; no exclusion applied)
19,CD82,0.632,0.773,0.477,1.04,1.03,0.971,0.625,0.0171,-0.0126,0.0197,-0.00451,0.0172,-0.279,-0.0863,0.00676,2,- (null not calibrated; no exclusion applied)
20,EVC2,0.624,0.853,0.0536,1.04,0,0.986,0.741,0.00502,0.00395,-0.0117,-0.0135,-0.0147,-0.392,0.0624,0.013,1,- (null not calibrated; no exclusion applied)
21,SLC5A3,0.617,0.893,0.217,1.11,1.33,1.14,0.747,0.00257,-0.0315,0.0253,-0.00454,-0.0376,-0.184,-0.0488,0.011,0,- (null not calibrated; no exclusion applied)
22,SGCB,0.615,0.867,0.0735,1.19,0,0.786,0.74,0.00384,0.00898,0.0122,0.00909,0.0505,0.475,0.0273,0.013,1,- (null not calibrated; no exclusion applied)
23,IL6R,0.614,0.72,0.233,1.12,1.49,0.777,0.933,-0.0154,0.00829,-0.00528,-0.00946,-0.0622,-0.333,0.289,0.00756,0,- (null not calibrated; no exclusion applied)
24,SYPL1,0.599,0.907,0.362,1.05,1.07,0.896,0.608,-0.0101,0.0096,-0.000845,0.014,0.0295,0.22,0.163,0.00582,1,- (null not calibrated; no exclusion applied)
25,CALU,0.587,0.88,0.19,1.15,2,1.01,0.561,-0.00041,-0.000574,0.00121,0.00185,0.057,0.358,0.155,0.00852,1,- (null not calibrated; no exclusion applied)
26,QSOX1,0.584,0.853,0.133,1.18,1.46,1.07,0.557,-0.0185,-0.00107,0.0205,0.0118,0.0523,0.353,-0.163,0.00862,1,- (null not calibrated; no exclusion applied)
27,VDAC1,0.584,0.947,0.482,1.02,1.19,0.976,0.67,0.0149,-0.00186,-0.00436,0.00861,0.135,1.17,-0.136,0.00527,2,- (null not calibrated; no exclusion applied)
28,AP2S1,0.579,0.907,0.358,1.08,1.15,0.862,0.555,0.00102,0.0285,-0.0162,0.0141,0.105,1.07,-0.214,0.00687,1,- (null not calibrated; no exclusion applied)
29,STX6,0.578,0.893,0.145,1.11,1.46,0.873,0.566,-0.0174,-0.00674,-0.00186,0.000462,0.0123,0.0281,-0.0322,0.0103,1,- (null not calibrated; no exclusion applied)
30,TSPAN3,0.574,0.84,0.0966,1.12,0,0.849,0.805,-0.0175,0.0129,0.0139,0.0157,0.069,0.464,0.112,0.0109,1,- (null not calibrated; no exclusion applied)
31,SLC39A14,0.566,0.853,0.057,1.1,0,0.979,0.758,0.0134,-0.00116,0.00847,0.000575,0.0472,0.248,0.00902,0.0198,2,- (null not calibrated; no exclusion applied)
32,SURF4,0.552,0.72,0.432,1.04,1.07,1.03,0.504,-0.00433,-0.0171,0.00271,-0.0103,0.0322,-0.0524,-0.173,0.00617,1,- (null not calibrated; no exclusion applied)
33,TNFRSF4,0.542,0.667,0.349,1.1,1.2,1.08,0.706,0.093,-0.0299,0.00607,-0.0134,-0.0304,-0.389,0.0926,0.0108,1,- (null not calibrated; no exclusion applied)
34,TSPAN2,0.53,0.747,0.135,0.952,2,0.664,0.677,0.00922,-0.00383,-0.00514,0.00935,-0.0129,-0.0812,0.07,0.0145,1,- (null not calibrated; no exclusion applied)
35,PTH2R,0.53,0.667,0.0304,0.85,0,0.546,0.434,-0.129,-0.0127,-0.000262,-0.0193,-0.00357,-0.374,0.119,0.02,0,- (null not calibrated; no exclusion applied)
36,QSOX2,0.512,0.853,0.0619,1.12,0,0.957,0.965,-0.0574,-0.00658,0.00802,-0.0142,-0.0155,-0.107,0.0888,0.0157,0,- (null not calibrated; no exclusion applied)
37,GPI,0.507,0.827,0.474,1.03,1.16,0.966,0.636,0.0175,-0.000104,0.00217,0.00607,0.0739,0.48,0.0193,0.00679,2,- (null not calibrated; no exclusion applied)
38,TMEM30A,0.506,0.787,0.188,1.07,1.34,0.724,0.331,-0.00649,-0.00363,-0.00342,-0.0138,-0.0219,-0.386,-0.116,0.00805,0,- (null not calibrated; no exclusion applied)
39,APP,0.495,0.587,0.14,1.24,1.79,0.649,0.772,0.019,0.00975,0.0157,0.0017,-0.0281,-0.207,0.0353,0.0116,1,- (null not calibrated; no exclusion applied)
40,MBP,0.493,0.773,0.616,1.02,1.06,0.994,0.574,-0.029,-0.0261,-0.000272,-0.0141,-0.155,-0.55,0.189,0.00451,0,- (null not calibrated; no exclusion applied)
41,NEO1,0.492,0.8,0.036,1.11,0,0.846,0.652,-0.0495,-0.00104,0.0027,-0.00667,-0.00704,0.0902,0.0439,0.014,1,- (null not calibrated; no exclusion applied)
42,FAM174B,0.491,0.747,0.0409,0.888,0,0.946,0.585,0.0704,0.000409,0.00833,-0.0117,-0.00185,0.067,0.0337,0.0156,2,- (null not calibrated; no exclusion applied)
43,TMED9,0.482,0.813,0.397,1.04,1.12,0.832,0.4,-0.00466,-0.00119,0.00784,-0.00242,0.056,0.142,-0.0229,0.00727,1,- (null not calibrated; no exclusion applied)
44,SIRPG,0.48,0.64,0.239,1.09,1.1,0.581,0.703,-0.00378,0.0129,-8.89e-05,0.00785,-0.0155,-0.0566,0.12,0.00973,1,- (null not calibrated; no exclusion applied)
45,CAPNS1,0.479,0.813,0.194,1.01,1.64,0.953,0.246,-0.00826,0.0158,0.00151,0.0119,0.08,0.39,-0.134,0.00705,1,- (null not calibrated; no exclusion applied)
46,LRPAP1,0.478,0.893,0.356,1.04,1.17,0.855,0.535,-0.0148,-0.00269,0.00311,0.00344,0.0418,0.512,0.0454,0.00796,1,- (null not calibrated; no exclusion applied)
47,CLTC,0.473,0.853,0.254,1.01,1.27,0.919,0.674,0.0137,-0.00583,0.00797,-0.00174,0.00622,0.0743,-0.14,0.00729,2,- (null not calibrated; no exclusion applied)
48,IL9R,0.472,0.64,0.0693,1.18,0,0.969,0.4,0.0561,0.00239,-0.00213,-0.00382,0.0204,-0.124,0.179,0.0142,2,- (null not calibrated; no exclusion applied)
49,CD84,0.471,0.733,0.222,1.14,1.73,0.806,0.718,-0.00726,-5.23e-05,0.00134,-0.00814,-0.0878,-0.475,0.113,0.0103,0,- (null not calibrated; no exclusion applied)
50,SLC7A1,0.467,0.84,0.103,1.11,0,0.795,0.636,0.0271,-0.0156,0.00669,-0.0119,0.038,0.297,0.156,0.0095,2,- (null not calibrated; no exclusion applied)
51,SLC52A2,0.466,0.867,0.173,1.11,1.97,0.863,0.857,-0.00208,0.00719,0.0219,-0.0018,0.0563,0.195,-0.0858,0.00819,1,- (null not calibrated; no exclusion applied)
52,SEC61G,0.461,0.853,0.539,1.09,1.16,0.895,0.769,0.00129,0.0423,-0.0163,0.015,0.0662,0.936,-0.042,0.00631,1,- (null not calibrated; no exclusion applied)
53,RALA,0.457,0.893,0.305,1.03,1.28,0.816,0.321,0.00255,0.00865,0.00789,-0.00581,0.0494,0.244,-0.0239,0.00719,1,- (null not calibrated; no exclusion applied)
54,SDC4,0.453,0.747,0.0859,1.36,0,1.34,0.81,0.0464,-0.0218,0.0289,0.000403,0.0625,0.523,-0.243,0.0128,2,- (null not calibrated; no exclusion applied)
55,SLC12A2,0.449,0.84,0.0787,1.16,0,0.913,0.501,0.0271,-0.0223,-0.00165,-0.00318,-0.0106,-0.0529,0.0562,0.0117,2,- (null not calibrated; no exclusion applied)
56,CHMP2A,0.448,0.84,0.315,1.01,1.19,0.665,0.521,-0.0056,0.0214,0.0172,0.0159,0.073,0.658,-0.174,0.00623,1,- (null not calibrated; no exclusion applied)
57,CHMP5,0.441,0.853,0.254,1.01,1.36,0.622,0.47,-4.83e-05,0.0121,0.000925,0.00215,0.0469,-0.00951,-0.103,0.00601,1,- (null not calibrated; no exclusion applied)
58,PDIA6,0.44,0.8,0.419,1.03,1.11,0.922,0.494,0.0223,-0.00294,0.00601,0.00919,0.0985,0.631,-0.0927,0.00686,2,- (null not calibrated; no exclusion applied)
59,NRROS,0.438,0.667,0.101,1.31,0,0.255,0.452,0.0311,-0.00317,0.00263,0.00493,0.0108,0.213,0.163,0.0119,3,- (null not calibrated; no exclusion applied)
60,P4HB,0.436,0.787,0.478,1.04,1.11,0.826,0.552,0.0068,-0.00194,0.0105,0.00357,0.0675,0.573,-0.125,0.00659,2,- (null not calibrated; no exclusion applied)
61,SLAMF1,0.432,0.653,0.323,1.05,1.2,0.856,0.847,-0.00158,-0.00686,-0.0148,-0.0203,-0.0809,-0.443,-0.0429,0.0093,0,- (null not calibrated; no exclusion applied)
62,C1QBP,0.43,0.8,0.353,1.01,1.08,0.552,0.712,0.0314,0.0174,-0.00466,0.00354,0.123,1.33,0.0413,0.00751,2,- (null not calibrated; no exclusion applied)
63,CHRNA6,0.429,0.68,0.0315,0.773,0,0.482,0.509,0.0224,-0.00796,-0.00367,-0.000414,0.0269,0.174,0.0662,0.0213,2,- (null not calibrated; no exclusion applied)
64,ANXA2,0.428,0.733,0.618,1.01,1.03,0.989,0.481,0.00802,0.0264,0.00201,0.00335,0.0694,1.17,-0.0542,0.00411,2,- (null not calibrated; no exclusion applied)
65,CLPTM1,0.427,0.84,0.191,1.06,1.57,0.948,0.399,-0.0133,0.00235,-0.00542,0.00505,0.0478,0.158,-0.11,0.00652,1,- (null not calibrated; no exclusion applied)
66,MLEC,0.426,0.84,0.235,1.02,1.36,0.565,0.464,-0.00746,-0.000909,0.0153,-0.00586,0.0743,0.609,-0.0463,0.00843,1,- (null not calibrated; no exclusion applied)
67,HSPD1,0.422,0.827,0.606,1.06,1.11,0.823,0.673,0.0264,-0.0409,0.0077,0.00208,0.033,1.03,0.123,0.00613,2,- (null not calibrated; no exclusion applied)
68,ENTPD1,0.42,0.6,0.137,1.01,0,0.273,0.311,0.0119,-0.0157,0.00304,-0.0182,-0.0173,0.0156,-0.1,0.0147,3,- (null not calibrated; no exclusion applied)
69,NPTN,0.419,0.893,0.152,1.09,0.785,0.955,0.333,-0.0229,0.00367,0.00575,-0.00356,0.0193,0.223,-0.106,0.0081,1,- (null not calibrated; no exclusion applied)
70,CASD1,0.414,0.84,0.074,1.18,0,1.12,0.607,-0.0113,0.000852,-0.00378,0.00206,-0.00869,-0.118,0.144,0.00899,0,- (null not calibrated; no exclusion applied)
71,MGAT4B,0.409,0.84,0.0964,1.15,0,1.02,0.783,-0.0263,0.00193,0.00474,-0.00107,0.0309,0.227,0.0938,0.0123,1,- (null not calibrated; no exclusion applied)
72,P2RY11,0.405,0.76,0.0531,0.783,0,0.559,0.562,0.0134,0.0111,0.0151,-0.00294,-0.00101,-0.00427,0.112,0.0208,2,- (null not calibrated; no exclusion applied)
73,SAR1B,0.404,0.867,0.228,1.12,1.42,0.778,0.617,-0.0197,-0.000886,-0.00989,-0.00274,0.0408,0.311,-0.0785,0.0104,1,- (null not calibrated; no exclusion applied)
74,CD109,0.403,0.68,0.0586,1.33,0,0.877,0.724,-0.00535,0.000197,0.000753,-0.0039,0.0185,0.117,-0.166,0.0163,1,- (null not calibrated; no exclusion applied)
75,AP2M1,0.403,0.88,0.389,1.07,1.25,0.88,0.302,-0.00503,0.0131,0.00299,0.023,0.0758,0.801,-0.14,0.00599,1,- (null not calibrated; no exclusion applied)
76,FGL2,0.4,0.627,0.0861,1.05,0,1.48,0.515,0.0636,-0.00357,0.00606,-0.00194,-0.0399,-0.325,0.129,0.0111,1,- (null not calibrated; no exclusion applied)
77,TMED4,0.4,0.867,0.363,1.05,1.17,0.789,0.331,-0.0149,-0.000798,0.00359,0.00141,-0.00771,-0.0431,0.251,0.00647,1,- (null not calibrated; no exclusion applied)
78,PRSS21,0.399,0.587,0.0209,0.316,0,0.851,0.377,-0.38,0.0152,0.0103,0.0163,0.0753,0.184,0.235,0.0148,2,- (null not calibrated; no exclusion applied)
79,LPAR5,0.399,0.707,0.0557,0.991,0,1.07,0.708,-0.0314,-0.00367,0.00149,0.00354,-0.00599,-0.13,0.165,0.0109,0,- (null not calibrated; no exclusion applied)
[440 further rows omitted from this block]
```

## §3 Candidates passing every gate

**What:** the genes that are selectively up on malignant CD4 and survived §2.

**How:** clean, q < 0.05, paired log2FC > 0.5, detected in >= 50% of donors, not a loss marker, not a lineage control.

```csv
index,gene,log2fc_paired,frac_donors_up,frac_donors_positive,median_pct_expr,q,z_blocked,retained_trim,retained_clean,retained_ds,rho_burden,worst_T_arm,cd_name,stage
0,CD70,1.1,0.92,0.808,0.21,3.09e-10,6.47,1.1,0.899,0.693,-0.0992,CD4_unassessed,CD70,phase 1
1,CD9,1.08,0.733,0.718,0.25,7.09e-07,4.8,1.13,0.987,0.906,-0.128,CD4_unassessed,,
2,CPM,1.02,0.747,0.577,0.137,7.08e-08,4.05,1.11,0.869,0.792,-0.081,CD4_unassessed,,
5,COL6A3,0.811,0.88,0.564,0.113,3.28e-09,5.01,1.03,1.15,0.934,0.0646,CD4_Treg,,
6,CD151,0.794,0.76,0.795,0.196,9.39e-08,5.11,1.15,0.943,0.791,0.0725,CD4_unassessed,,
8,GRN,0.778,0.907,0.846,0.211,9.81e-10,5.87,1.08,0.886,0.602,0.0709,CD4_unassessed,,
11,TFRC,0.731,0.893,0.949,0.296,1.67e-09,4.85,1.07,1.06,0.706,0.0715,CD4_unassessed,,
12,SLC39A6,0.707,0.947,0.91,0.183,1.91e-10,6.98,1.15,1.08,0.599,-0.0683,CD4_unassessed,,
13,SLC4A2,0.698,0.933,0.641,0.13,1.91e-10,6.44,1.09,1.1,0.819,-0.000114,CD4_unassessed,,
14,TNFRSF25,0.698,0.747,0.872,0.339,9.14e-05,2.92,1.05,0.855,0.77,0.251,CD4_unassessed,,
15,SLC1A4,0.69,0.747,0.628,0.144,5.23e-05,3.94,1.09,0.66,0.534,-0.179,CD4_unassessed,,
16,LGALS3,0.663,0.587,0.885,0.427,0.00115,2.72,1.07,0.861,0.917,-0.0744,CD4_unassessed,,
17,PDIA4,0.637,0.88,0.859,0.255,2.5e-09,3.77,1.11,0.901,0.595,-0.0762,CD4_unassessed,,
18,FUT8,0.635,0.867,0.744,0.184,6.88e-09,4.14,1.1,1.03,0.692,-0.0842,CD4_unassessed,,
19,CD82,0.632,0.773,1,0.477,1.41e-07,4.19,1.04,0.971,0.625,-0.0863,CD4_unassessed,,
21,SLC5A3,0.617,0.893,0.769,0.217,1.81e-09,4.38,1.11,1.14,0.747,-0.0488,CD4_unassessed,,
23,IL6R,0.614,0.72,0.769,0.233,2.31e-05,2.87,1.12,0.777,0.933,0.289,CD4_unassessed,,
24,SYPL1,0.599,0.907,0.949,0.362,1.15e-09,4.32,1.05,0.896,0.608,0.163,CD4_unassessed,,
25,CALU,0.587,0.88,0.846,0.19,6.18e-10,6.3,1.15,1.01,0.561,0.155,CD4_unassessed,,
26,QSOX1,0.584,0.853,0.692,0.133,7.51e-09,5.22,1.18,1.07,0.557,-0.163,CD4_unassessed,,
27,VDAC1,0.584,0.947,0.987,0.482,1.91e-10,5.4,1.02,0.976,0.67,-0.136,CD4_unassessed,,
28,AP2S1,0.579,0.907,0.91,0.358,4.57e-10,4.21,1.08,0.862,0.555,-0.214,CD4_unassessed,,
29,STX6,0.578,0.893,0.679,0.145,5.02e-09,5.17,1.11,0.873,0.566,-0.0322,CD4_unassessed,,
32,SURF4,0.552,0.72,0.974,0.432,6.92e-07,5.17,1.04,1.03,0.504,-0.173,CD4_unassessed,,
33,TNFRSF4,0.542,0.667,0.897,0.349,0.00308,1.73,1.1,1.08,0.706,0.0926,CD4_Treg,,
34,TSPAN2,0.53,0.747,0.654,0.135,1.24e-05,2.78,0.952,0.664,0.677,0.07,CD4_reactive,,
37,GPI,0.507,0.827,0.974,0.474,2.79e-07,3.22,1.03,0.966,0.636,0.0193,CD4_unassessed,,
38,TMEM30A,0.506,0.787,0.846,0.188,1.08e-08,2.85,1.07,0.724,0.331,-0.116,CD4_unassessed,,
```

## §3 Loss markers

**What:** the CTCL loss phenotype, as the sanity check that cleaning did not flatten real biology.

**How:** same paired test.

```csv
index,gene,log2fc_paired,frac_donors_up,q,artifact_flag
457,CD6,0.0957,0.56,0.344,clean
2062,CD5,-0.0584,0.493,0.816,clean
2152,CD2,-0.146,0.453,0.0478,clean
2255,CD27,-0.359,0.467,0.792,clean
2325,DPP4,-0.841,0.253,0.00391,clean
2354,CD7,-2.85,0.0933,3.6e-09,clean
```

## §3 FD Per-gene summary of the donor x gene effect matrix

**What:** how consistent each leading antigen's effect is across patients.

**How:** per-donor log2FC matrix; frac_up_1 is the share of donors where the antigen is at least doubled.

```csv
gene,n_donors,median,iqr,min,max,frac_up,frac_up_1
CD70,75,1.1,1.45,-0.849,3.33,0.92,0.56
CD9,75,1.08,2.4,-2.33,5.31,0.733,0.547
CPM,75,1.02,1.79,-1.89,3.55,0.747,0.507
COL6A3,75,0.811,1.19,-1.26,3.48,0.88,0.427
CD151,75,0.794,1.28,-1.11,2.61,0.76,0.44
GRN,75,0.778,0.756,-1.03,2.57,0.907,0.333
TFRC,75,0.731,0.728,-0.887,2.93,0.893,0.347
SLC39A6,75,0.707,0.731,-0.293,2.66,0.947,0.307
SLC4A2,75,0.698,0.82,-0.762,2.34,0.933,0.36
TNFRSF25,75,0.698,1.29,-3.51,2.37,0.747,0.307
SLC1A4,75,0.69,1.3,-2.1,2.83,0.747,0.307
LGALS3,75,0.663,2.14,-3.76,2.86,0.587,0.48
```

## §3 FE Eligibility

**What:** in what fraction of patients at least X% of the malignant CD4 cells carry the antigen.

**How:** transcript detection per donor, thresholded.

**Full table on disk:** `tables/tgt42_eligibility_skin.csv`

```csv
index,gene,n_donors,median_pct_expr,frac_donors_ge10,frac_donors_ge25,frac_donors_ge50,frac_donors_ge75
10,TFRC,75,0.294,0.96,0.6,0.2,0.04
8,SLC39A6,75,0.187,0.92,0.293,0.0667,0
11,TNFRSF25,75,0.328,0.88,0.613,0.28,0.04
6,LGALS3,75,0.48,0.88,0.773,0.48,0.253
5,GRN,75,0.212,0.867,0.453,0.107,0
0,CD151,75,0.2,0.827,0.427,0.107,0.0133
1,CD70,75,0.215,0.813,0.427,0.0533,0
2,CD9,75,0.262,0.733,0.507,0.227,0.0267
9,SLC4A2,75,0.119,0.64,0.2,0.0267,0
7,SLC1A4,75,0.143,0.627,0.307,0.0533,0.0133
4,CPM,75,0.133,0.573,0.24,0.0533,0.0133
3,COL6A3,75,0.111,0.56,0.173,0.107,0.0267
```

## §4 Off-tumour

**What:** what else carries each candidate: the worst non-target T arm, healthy CD4, the worst non-T skin compartment and healthy skin.

**How:** T arms from this screen's long table (share of cells detected); non-T compartments from the degradome pseudobulk cube (log1p CP10K, sample level, median per level); healthy skin from Reynolds 2021, which is unblocked and flags rather than ranks.

```csv
index,gene,worst_T_arm,worst_T_arm_pct,healthy_t_burden,worst_nonT_skin,worst_nonT_level,worst_healthy_skin,worst_healthy_skin_level
0,CD70,CD4_unassessed,0.331,0.024,0.77,B,0,CD4
1,CD9,CD4_unassessed,0.455,0.107,2.7,F_papillary,,
2,CPM,CD4_unassessed,0.272,0.00669,1.15,Mast,,
5,COL6A3,CD4_Treg,0.0385,0.0158,3.03,F_papillary,0.902,Fibroblast
6,CD151,CD4_unassessed,0.32,0.0612,1.74,F_myofibroblast,,
8,GRN,CD4_unassessed,0.368,0.041,2.59,Mac_SPP1_TREM2,,
11,TFRC,CD4_unassessed,0.537,0.118,1.09,Mac_SPP1_TREM2,,
12,SLC39A6,CD4_unassessed,0.49,0.0447,1,Melanocyte,,
13,SLC4A2,CD4_unassessed,0.402,0.0156,0.371,Melanocyte,,
14,TNFRSF25,CD4_unassessed,0.497,0.0943,0.221,Neutrophils,,
15,SLC1A4,CD4_unassessed,0.414,0.0239,1.23,Melanocyte,,
16,LGALS3,CD4_unassessed,0.635,0.361,3.24,Mast,3.69,Mast
17,PDIA4,CD4_unassessed,0.439,0.082,0.878,Plasma,,
18,FUT8,CD4_unassessed,0.415,0.0373,0.38,B,,
19,CD82,CD4_unassessed,0.574,0.176,1.58,Mast,,
21,SLC5A3,CD4_unassessed,0.513,0.0499,0.843,F_mesenchymal,,
23,IL6R,CD4_unassessed,0.373,0.0726,1.12,Neutrophils,0,CD4
24,SYPL1,CD4_unassessed,0.62,0.179,1.29,B,,
25,CALU,CD4_unassessed,0.456,0.0392,1.48,F_mesenchymal,,
26,QSOX1,CD4_unassessed,0.366,0.0261,0.862,Neutrophils,,
27,VDAC1,CD4_unassessed,0.708,0.175,1.2,Melanocyte,,
28,AP2S1,CD4_unassessed,0.456,0.0957,1.68,Mono_CD14,,
29,STX6,CD4_unassessed,0.314,0.0261,0.402,Neutrophils,,
32,SURF4,CD4_unassessed,0.749,0.185,0.836,F_mesenchymal,,
33,TNFRSF4,CD4_Treg,0.556,0.137,0.639,pDC,1.17,moDC
34,TSPAN2,CD4_reactive,0.0534,0.0377,0.259,Vascular,,
37,GPI,CD4_unassessed,0.813,0.125,0.977,Mac_infl,,
38,TMEM30A,CD4_unassessed,0.421,0.0612,0.625,F_papillary,,
```

## §5 Trafficking modules per arm

**What:** the state of the machinery that puts a protein on the surface and keeps it there.

**How:** per-module z across donors, then the arm mean. Context, not a correction factor.

```csv
module,CD4_malignant,CD4_reactive,CD4_HC
endocytosis,0.342,-0.0524,-1.43
er_folding_qc,0.415,-0.102,-1.11
er_to_golgi,0.452,-0.128,-1.26
er_translocation,0.339,-0.064,-1.17
escrt_exosome,0.492,-0.173,-1.39
glycosylation,0.253,0.0391,-0.931
recycling,0.321,-0.0376,-1.34
retrograde,0.41,-0.046,-1.31
shedding,0.423,0.0964,-0.63
snare_fusion,0.251,-0.11,-1.28
ubiquitin_proteostasis,0.163,-0.02,-1.19
```

## §6 Escape risk

**What:** how much each candidate's detection varies between patients and between one patient's own clusters.

**How:** SD of detection across donors and, within a donor, across its per-sample clusters.

```csv
index,gene,between_donor_sd,within_donor_sd,min_donor_pct
114,AP2S1,0.187,0.203,0.0222
228,CALU,0.134,0.129,0.0256
256,CD151,0.164,0.129,0.0171
313,CD70,0.147,0.126,0.00494
319,CD82,0.2,0.175,0.114
325,CD9,0.225,0.134,0
471,COL6A3,0.209,0.0758,0
478,CPM,0.175,0.0798,0
718,FUT8,0.199,0.0868,0
807,GPI,0.212,0.176,0.0476
888,GRN,0.161,0.129,0.0213
1003,IL6R,0.15,0.0797,0
1117,LGALS3,0.289,0.139,0.0156
1637,PDIA4,0.174,0.145,0.0159
1745,QSOX1,0.122,0.0772,0.00285
1928,SLC1A4,0.182,0.126,0.00386
1976,SLC39A6,0.135,0.113,0.0316
1993,SLC4A2,0.127,0.0753,0.0102
2004,SLC5A3,0.172,0.107,0.0209
2104,STX6,0.108,0.0749,0
2112,SURF4,0.201,0.179,0.0313
2117,SYPL1,0.18,0.145,0.0476
2138,TFRC,0.2,0.123,0.0293
2196,TMEM30A,0.123,0.0925,0.0256
2227,TNFRSF25,0.22,0.121,0.00184
2228,TNFRSF4,0.228,0.177,0
2268,TSPAN2,0.12,0.0752,0.00227
2315,VDAC1,0.192,0.204,0.0805
```

## §7 Control check

**What:** where the antigens with a known answer sit in the ranking. This decides whether anything else here is evidence.

**How:** rank by paired log2FC over all screened genes.

```csv
index,gene,rank,log2fc_paired,frac_donors_positive
129,TNFRSF8,130,0.332,0.192
136,CCR4,137,0.324,0.705
706,KIR3DL2,690,0,0.321
2179,IL2RA,2180,-0.172,0.628
```

## §7 Shortlist with the plausibility layer

**What:** the final candidate table: effect, prevalence, off-tumour cost, heterogeneity, clinical precedent and surfaceome evidence.

**How:** surface_confident / n_tm / surfy_class come from CSPA and Surfy, not from this cohort; they are the check that a statistically clean hit is a protein an antibody could reach.

**Full table on disk:** `tables/tgt42_shortlist_skin.csv`

```csv
index,gene,cd_name,frac_donors_positive,median_pct_expr,log2fc_paired,q,worst_T_arm,healthy_t_burden,worst_nonT_skin,worst_nonT_level,worst_healthy_skin,between_donor_sd,stage,n_evidence,surface_confident,n_tm,surfy_class,surface_tier,artifact_flag
0,CD70,CD70,0.808,0.21,1.1,3.09e-10,CD4_unassessed,0.024,0.77,B,0,0.147,phase 1,2,True,1.0,Unclassified,unknown,clean
1,CD9,,0.718,0.25,1.08,7.09e-07,CD4_unassessed,0.107,2.7,F_papillary,,0.225,,3,True,4.0,Miscellaneous,unknown,clean
2,CPM,,0.577,0.137,1.02,7.08e-08,CD4_unassessed,0.00669,1.15,Mast,,0.175,,3,True,0.0,Enzymes,unknown,clean
5,COL6A3,,0.564,0.113,0.811,3.28e-09,CD4_Treg,0.0158,3.03,F_papillary,0.902,0.209,,1,True,,,unknown,clean
6,CD151,,0.795,0.196,0.794,9.39e-08,CD4_unassessed,0.0612,1.74,F_myofibroblast,,0.164,,3,True,4.0,Miscellaneous,unknown,clean
8,GRN,,0.846,0.211,0.778,9.81e-10,CD4_unassessed,0.041,2.59,Mac_SPP1_TREM2,,0.161,,1,True,,,unknown,clean
11,TFRC,,0.949,0.296,0.731,1.67e-09,CD4_unassessed,0.118,1.09,Mac_SPP1_TREM2,,0.2,,2,True,,,unknown,clean
12,SLC39A6,,0.91,0.183,0.707,1.91e-10,CD4_unassessed,0.0447,1,Melanocyte,,0.135,,4,True,6.0,Transporters,unknown,clean
13,SLC4A2,,0.641,0.13,0.698,1.91e-10,CD4_unassessed,0.0156,0.371,Melanocyte,,0.127,,2,True,,,unknown,clean
14,TNFRSF25,,0.872,0.339,0.698,9.14e-05,CD4_unassessed,0.0943,0.221,Neutrophils,,0.22,,3,True,1.0,Receptors,unknown,clean
15,SLC1A4,,0.628,0.144,0.69,5.23e-05,CD4_unassessed,0.0239,1.23,Melanocyte,,0.182,,3,True,9.0,Transporters,unknown,clean
16,LGALS3,,0.885,0.427,0.663,0.00115,CD4_unassessed,0.361,3.24,Mast,3.69,0.289,,2,True,,,unknown,clean
17,PDIA4,,0.859,0.255,0.637,2.5e-09,CD4_unassessed,0.082,0.878,Plasma,,0.174,,,,,,unknown,clean
18,FUT8,,0.744,0.184,0.635,6.88e-09,CD4_unassessed,0.0373,0.38,B,,0.199,,,,,,unknown,clean
19,CD82,,1,0.477,0.632,1.41e-07,CD4_unassessed,0.176,1.58,Mast,,0.2,,2,True,4.0,Miscellaneous,unknown,clean
21,SLC5A3,,0.769,0.217,0.617,1.81e-09,CD4_unassessed,0.0499,0.843,F_mesenchymal,,0.172,,3,True,12.0,Transporters,unknown,clean
23,IL6R,,0.769,0.233,0.614,2.31e-05,CD4_unassessed,0.0726,1.12,Neutrophils,0,0.15,,2,True,1.0,Receptors,unknown,clean
24,SYPL1,,0.949,0.362,0.599,1.15e-09,CD4_unassessed,0.179,1.29,B,,0.18,,2,True,4.0,Miscellaneous,unknown,clean
25,CALU,,0.846,0.19,0.587,6.18e-10,CD4_unassessed,0.0392,1.48,F_mesenchymal,,0.134,,1,True,,,unknown,clean
26,QSOX1,,0.692,0.133,0.584,7.51e-09,CD4_unassessed,0.0261,0.862,Neutrophils,,0.122,,2,True,1.0,Enzymes,unknown,clean
27,VDAC1,,0.987,0.482,0.584,1.91e-10,CD4_unassessed,0.175,1.2,Melanocyte,,0.192,,1,True,,,unknown,clean
28,AP2S1,,0.91,0.358,0.579,4.57e-10,CD4_unassessed,0.0957,1.68,Mono_CD14,,0.187,,,,,,unknown,clean
29,STX6,,0.679,0.145,0.578,5.02e-09,CD4_unassessed,0.0261,0.402,Neutrophils,,0.108,,,,,,unknown,clean
32,SURF4,,0.974,0.432,0.552,6.92e-07,CD4_unassessed,0.185,0.836,F_mesenchymal,,0.201,,,,,,unknown,clean
33,TNFRSF4,,0.897,0.349,0.542,0.00308,CD4_Treg,0.137,0.639,pDC,1.17,0.228,,2,True,1.0,Receptors,unknown,clean
34,TSPAN2,,0.654,0.135,0.53,1.24e-05,CD4_reactive,0.0377,0.259,Vascular,,0.12,,2,True,4.0,Miscellaneous,unknown,clean
37,GPI,,0.974,0.474,0.507,2.79e-07,CD4_unassessed,0.125,0.977,Mac_infl,,0.212,,2,True,,,unknown,clean
38,TMEM30A,,0.846,0.188,0.506,1.08e-08,CD4_unassessed,0.0612,0.625,F_papillary,,0.123,,2,True,2.0,Miscellaneous,unknown,clean
```

======================== COPY TO HERE ==========================