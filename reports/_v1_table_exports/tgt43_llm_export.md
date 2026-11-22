> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 43 — Surface-target screen, blood

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**, from blood_T_annotated.h5ad
(10,000 genes in the object -- an HVG SUBSET, so a gene absent from the
screen is not necessarily unexpressed; CD4 and TNFRSF8/CD30 are both missing for this reason).

Cohort: 131 donor keys reduced to 75 independent keys by
dropping the geskin2026 ex vivo drug-culture wells and duplicate persons, and by moving five
healthy donors that the atlas labelled SS out of the reactive arm. Diagnosis from the curated
`clin_dx`. 52 donors carry both a malignant and a reactive CD4 arm.

Arms: malignancy call `mal_cluster`. Blood in Sezary syndrome is ~92% malignant
CD4, so the reactive arm is small in many donors. The CD8 arm is a CD8+NK mixture (nb31 labelled
on CD8A) and any statement about it is a statement about a mixture.

Headline test: per gene, the within-donor difference of mean log1p CP10K between the malignant
and reactive arms, paired Wilcoxon over donors, BH-adjusted (`q`). `log2fc_paired` is the median
per-donor log2 ratio; `frac_donors_up` the share of donors with a positive difference.

Artifact tests: six, against an empirical null made of transcripts a CD4 T cell cannot make
(LYZ, S100A8/9, CD14, VCAN, FCN1, CST3, LST1, AIF1, TYROBP, FCER1G, CD68, MNDA, PPBP, PF4, HBB,
MS4A1, CD79A). Thresholds are read off that null rather than chosen, and the section is allowed
to exclude nothing: the calibration verdict decides. Verdict here: 18 lineage controls: median log2FC +0.00, median frac_donors_up 0.39 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.

Known answers, for calibration: CCR4 (mogamulizumab) is approved in CTCL, KIR3DL2 is lacutamab's
target, IL2RA is registrational; TNFRSF8/CD30 is absent from this object. CD7 and DPP4/CD26 loss
is the Sezary immunophenotype and must come out negative.

Scope limits that apply to every number: the off-tumour axis covers T and NK arms ONLY -- the
non-T blood compartments are not in the screen table, so a myeloid or HSPC hit, the kind that
causes cytopenia, is invisible here. Nothing below has been seen as protein; notebook 45 does
that for 17 Sezary donors with CITE-seq.

## §0b Cohort

**What:** the independent donor set and its clinical fields.

**How:** curated clin_* columns only; no expression used.

**Full table on disk:** `tables/tgt43_blood_cohort.csv`

```csv
donor,study,dx,dx2,is_cultured,clone_shared,sex,age,n_cells,burden
B1__P4,borcherding2019,SS,SS,False,False,M,61,6260,0.982
B2__CTCL2,borcherding2023,SS,SS,False,False,F,71,31441,
B2__CTCL3,borcherding2023,SS,SS,False,False,M,67,2015,
B2__CTCL4,borcherding2023,SS,SS,False,False,F,84,305,0.577
B2__CTCL5,borcherding2023,SS,SS,False,False,M,73,1040,0.739
B2__CTCL6,borcherding2023,SS,SS,False,False,F,69,1829,0.495
B4__PF46_HTO5,geskin2026,HC,HC,True,False,F,,49,
B4__PF47_HTO1,geskin2026,HC,HC,True,False,M,,3116,
B4__PF48_HTO1,geskin2026,HC,HC,True,False,M,,1922,
B4__PF49_HTO1,geskin2026,HC,HC,True,False,F,,2784,
B4__PF50_HTO1,geskin2026,HC,HC,True,False,M,,5711,
B4__SZ30_HTO6,geskin2026,MF/SS,unresolved,True,False,M,70,2578,0.646
B4__SZ32_HTO1,geskin2026,SS,SS,True,False,M,32,2582,0.142
B4__SZ33_HTO5,geskin2026,SS,SS,True,False,M,,261,
B4__SZ38_HTO1,geskin2026,SS,SS,True,False,M,,4882,0.994
B4__SZ39_HTO1,geskin2026,SS,SS,True,False,F,,1489,0.557
B4__SZ5_HTO1,geskin2026,SS,SS,True,False,F,,3013,0.985
B5__PT01,buus2025,SS,SS,False,False,M,76,3395,
B5__PT02,buus2025,SS,SS,False,False,M,,5832,0.993
B5__PT04,buus2025,SS,SS,False,False,M,61,527,
B5__PT05,buus2025,SS,SS,False,True,F,,5017,0.979
B5__PT06,buus2025,SS,SS,False,True,F,79,5856,0.974
B5__PT07,buus2025,SS,SS,False,False,F,60,2624,0.973
B5__PT08,buus2025,SS,SS,False,False,M,75,4701,0.969
B5__PT09,buus2025,SS,SS,False,False,F,71,3681,0.946
B5__PT10,buus2025,SS,SS,False,False,M,79,1897,0.953
B5__PT13,buus2025,SS,SS,False,False,M,80,624,0.935
B5__PT14,buus2025,SS,SS,False,False,M,57,7348,0.933
B5__PT15,buus2025,SS,SS,False,False,M,,629,0.91
B5__PT16,buus2025,SS,SS,False,False,M,67,4713,0.961
B5__PT18,buus2025,MF,MF,False,True,F,,3895,0.791
B5__PT21,buus2025,MF,MF,False,False,M,70,3531,0.296
B5__PT22,buus2025,SS,SS,False,False,M,70,1620,0.22
B5__PT23,buus2025,SS,SS,False,False,F,85,5325,0.333
B6__N1,ren2023,HC,HC,False,False,F,,11258,
B6__N2,ren2023,HC,HC,False,False,M,,12554,
B6__N3,ren2023,HC,HC,False,False,F,,10696,
B6__P1,ren2023,SS,SS,False,False,M,81,9252,0.979
B6__P10,ren2023,MF,MF,False,False,M,70,8086,0.707
B6__P11,ren2023,MF,MF,False,False,M,76,4899,0.784
B6__P2,ren2023,MF,MF,False,False,M,76,12184,0.745
B6__P3,ren2023,SS,SS,False,False,F,72,10189,0.901
B6__P4,ren2023,MF,MF,False,False,M,79,11927,0.714
B6__P5,ren2023,SS,SS,False,False,M,56,7150,0.723
B6__P6,ren2023,SS,SS,False,False,M,85,12065,0.954
B6__P7,ren2023,SS,SS,False,False,F,78,8162,0.817
B6__P8,ren2023,SS,SS,False,False,F,74,8167,
B6__P9,ren2023,SS,SS,False,False,F,68,8807,0.923
B7__MF1,harro2023,MF,MF,False,False,F,81,42688,
B7__MF2,harro2023,MF,MF,False,False,F,81,21447,
B7__MF3,harro2023,MF,MF,False,False,M,68,12959,
B7__SS1,harro2023,SS,SS,False,False,F,76,16935,0.908
B7__SS2,harro2023,SS,SS,False,False,M,80,7920,0.979
B7__SS3,harro2023,SS,SS,False,False,M,49,19634,
B7__SS4,harro2023,SS,SS,False,False,M,70,8210,0.927
B8__1084,dorando2026,MF_or_SS_unresolved,unresolved,False,False,F,74,23116,0.982
B8__1126,dorando2026,MF_or_SS_unresolved,unresolved,False,False,F,61,21654,0.941
B8__1135,dorando2026,MF_or_SS_unresolved,unresolved,False,False,M,49,7167,0.994
B8__1137,dorando2026,MF_or_SS_unresolved,unresolved,False,False,M,27,6695,
B8__1181,dorando2026,MF_or_SS_unresolved,unresolved,False,False,M,59,10275,0.1
B8__999,dorando2026,MF_or_SS_unresolved,unresolved,False,False,F,,8796,0.898
D13__HB1,gaydosik2022,HC,HC,False,False,F,,2862,
D13__HB2,gaydosik2022,HC,HC,False,False,F,,2294,
D13__HB3,gaydosik2022,HC,HC,False,False,F,,1732,
D13__SZ16,gaydosik2022,SS,SS,False,False,M,56,6427,0.05
D13__SZ22,gaydosik2022,MF/SS,unresolved,False,False,M,62,2066,0.319
D13__SZ29,gaydosik2022,SS,SS,False,False,M,68,7418,0.955
D5__MFIVB,rindler2021_fi,MF,MF,False,False,F,65,1393,0.419
H__HC1,herrera2021,HC,HC,False,False,M,41,2894,
H__MFIV1,herrera2021,MF,MF,False,False,F,39,4788,0.499
H__SS1,herrera2021,SS,SS,False,False,F,49,3421,0.601
H__SS3,herrera2021,SS,SS,False,False,F,71,1568,0.771
H__SS4,herrera2021,SS,SS,False,False,M,56,1707,0.33
H__SS5,herrera2021,SS,SS,False,False,F,71,2065,0.578
H__SS6,herrera2021,SS,SS,False,False,M,78,2517,0.599
```

## §0b Dropped keys

**What:** the keys removed, with the reason.

**How:** ex vivo drug-culture arm, or the same person under another key.

```csv
donor,study,dx,clin_ex_vivo,person,n_cells,drop_reason
B1__P5,borcherding2019,SS,none,B1__P4,7910,same person as B1__P4
B4__PF46_HTO10,geskin2026,HC,culture_drug,pittsburgh:PF46,3336,ex vivo drug-culture arm
B4__PF46_HTO6,geskin2026,HC,culture_drug,pittsburgh:PF46,3049,ex vivo drug-culture arm
B4__PF46_HTO7,geskin2026,HC,culture_drug,pittsburgh:PF46,3185,ex vivo drug-culture arm
B4__PF46_HTO8,geskin2026,HC,culture_drug,pittsburgh:PF46,4614,ex vivo drug-culture arm
B4__PF46_HTO9,geskin2026,HC,culture_drug,pittsburgh:PF46,2625,ex vivo drug-culture arm
B4__PF47_HTO2,geskin2026,HC,culture_drug,pittsburgh:PF47,3789,ex vivo drug-culture arm
B4__PF47_HTO3,geskin2026,HC,culture_drug,pittsburgh:PF47,2739,ex vivo drug-culture arm
B4__PF47_HTO4,geskin2026,HC,culture_drug,pittsburgh:PF47,2653,ex vivo drug-culture arm
B4__PF47_HTO5,geskin2026,HC,culture_drug,pittsburgh:PF47,1730,ex vivo drug-culture arm
B4__PF47_HTO6,geskin2026,HC,culture_drug,pittsburgh:PF47,3240,ex vivo drug-culture arm
B4__PF48_HTO2,geskin2026,HC,culture_drug,pittsburgh:PF48,2447,ex vivo drug-culture arm
B4__PF48_HTO3,geskin2026,HC,culture_drug,pittsburgh:PF48,1707,ex vivo drug-culture arm
B4__PF48_HTO4,geskin2026,HC,culture_drug,pittsburgh:PF48,1855,ex vivo drug-culture arm
B4__PF48_HTO5,geskin2026,HC,culture_drug,pittsburgh:PF48,1636,ex vivo drug-culture arm
B4__PF48_HTO6,geskin2026,HC,culture_drug,pittsburgh:PF48,1717,ex vivo drug-culture arm
B4__PF49_HTO2,geskin2026,HC,culture_drug,pittsburgh:PF49,5807,ex vivo drug-culture arm
B4__PF49_HTO3,geskin2026,HC,culture_drug,pittsburgh:PF49,1853,ex vivo drug-culture arm
B4__PF49_HTO4,geskin2026,HC,culture_drug,pittsburgh:PF49,1624,ex vivo drug-culture arm
B4__PF49_HTO5,geskin2026,HC,culture_drug,pittsburgh:PF49,2126,ex vivo drug-culture arm
B4__PF49_HTO6,geskin2026,HC,culture_drug,pittsburgh:PF49,2995,ex vivo drug-culture arm
B4__PF50_HTO2,geskin2026,HC,culture_drug,pittsburgh:PF50,5167,ex vivo drug-culture arm
B4__PF50_HTO3,geskin2026,HC,culture_drug,pittsburgh:PF50,2249,ex vivo drug-culture arm
B4__PF50_HTO4,geskin2026,HC,culture_drug,pittsburgh:PF50,1574,ex vivo drug-culture arm
B4__PF50_HTO5,geskin2026,HC,culture_drug,pittsburgh:PF50,1820,ex vivo drug-culture arm
B4__PF50_HTO6,geskin2026,HC,culture_drug,pittsburgh:PF50,4170,ex vivo drug-culture arm
B4__SZ30_HTO10,geskin2026,MF/SS,culture_drug,pittsburgh:SZ30,1446,ex vivo drug-culture arm
B4__SZ30_HTO7,geskin2026,MF/SS,culture_drug,pittsburgh:SZ30,2176,ex vivo drug-culture arm
B4__SZ30_HTO8,geskin2026,MF/SS,culture_drug,pittsburgh:SZ30,1636,ex vivo drug-culture arm
B4__SZ30_HTO9,geskin2026,MF/SS,culture_drug,pittsburgh:SZ30,1989,ex vivo drug-culture arm
B4__SZ32_HTO2,geskin2026,SS,culture_drug,pittsburgh:MF26,1621,ex vivo drug-culture arm
B4__SZ32_HTO3,geskin2026,SS,culture_drug,pittsburgh:MF26,1383,ex vivo drug-culture arm
B4__SZ32_HTO4,geskin2026,SS,culture_drug,pittsburgh:MF26,1881,ex vivo drug-culture arm
B4__SZ32_HTO5,geskin2026,SS,culture_drug,pittsburgh:MF26,785,ex vivo drug-culture arm
B4__SZ32_HTO6,geskin2026,SS,culture_drug,pittsburgh:MF26,2105,ex vivo drug-culture arm
B4__SZ33_HTO10,geskin2026,SS,culture_drug,pittsburgh:SZ33,1106,ex vivo drug-culture arm
B4__SZ33_HTO6,geskin2026,SS,culture_drug,pittsburgh:SZ33,877,ex vivo drug-culture arm
B4__SZ33_HTO7,geskin2026,SS,culture_drug,pittsburgh:SZ33,1726,ex vivo drug-culture arm
B4__SZ33_HTO8,geskin2026,SS,culture_drug,pittsburgh:SZ33,1624,ex vivo drug-culture arm
B4__SZ33_HTO9,geskin2026,SS,culture_drug,pittsburgh:SZ33,1067,ex vivo drug-culture arm
B4__SZ38_HTO2,geskin2026,SS,culture_drug,pittsburgh:SZ38,4203,ex vivo drug-culture arm
B4__SZ38_HTO3,geskin2026,SS,culture_drug,pittsburgh:SZ38,4758,ex vivo drug-culture arm
B4__SZ38_HTO4,geskin2026,SS,culture_drug,pittsburgh:SZ38,3400,ex vivo drug-culture arm
B4__SZ38_HTO5,geskin2026,SS,culture_drug,pittsburgh:SZ38,3716,ex vivo drug-culture arm
B4__SZ38_HTO6,geskin2026,SS,culture_drug,pittsburgh:SZ38,3981,ex vivo drug-culture arm
B4__SZ39_HTO2,geskin2026,SS,culture_drug,pittsburgh:SZ39,1820,ex vivo drug-culture arm
B4__SZ39_HTO3,geskin2026,SS,culture_drug,pittsburgh:SZ39,1348,ex vivo drug-culture arm
B4__SZ39_HTO4,geskin2026,SS,culture_drug,pittsburgh:SZ39,1279,ex vivo drug-culture arm
B4__SZ39_HTO5,geskin2026,SS,culture_drug,pittsburgh:SZ39,1260,ex vivo drug-culture arm
B4__SZ39_HTO6,geskin2026,SS,culture_drug,pittsburgh:SZ39,1940,ex vivo drug-culture arm
B4__SZ5_HTO2,geskin2026,SS,culture_drug,pittsburgh:SZ5,5142,ex vivo drug-culture arm
B4__SZ5_HTO3,geskin2026,SS,culture_drug,pittsburgh:SZ5,3933,ex vivo drug-culture arm
B4__SZ5_HTO4,geskin2026,SS,culture_drug,pittsburgh:SZ5,3767,ex vivo drug-culture arm
B4__SZ5_HTO5,geskin2026,SS,culture_drug,pittsburgh:SZ5,4120,ex vivo drug-culture arm
B4__SZ5_HTO6,geskin2026,SS,culture_drug,pittsburgh:SZ5,3416,ex vivo drug-culture arm
H__SS2,herrera2021,SS,none,pittsburgh:MF19,1926,same person as D13__SZ22
```

## §1 Arms

**What:** donors, cells and studies per arm after the cohort fix.

**How:** groups of >= 20 cells.

```csv
arm,donors,cells,studies
CD4_malignant,58,295656,10
CD4_reactive,57,111125,10
CD8,72,70525,10
CD4_HC,12,40400,4
NK,6,756,3
```

## §1 Contamination load per arm

**What:** how dirty each arm is -- the asymmetry a contamination artifact would need.

**How:** median over the arm's donors.

```csv
arm,frac_lineage_pos,mean_myeloid_score,median_total_counts,median_doublet_score
CD4_HC,0.0845,0.0845,4.02e+03,0.061
CD4_malignant,0.0177,0.014,4.16e+03,0.0702
CD4_reactive,0.0114,0.0333,4.19e+03,0.042
CD8,0.0189,0.104,3.68e+03,0.036
NK,0.075,0.422,4.15e+03,0.0274
```

## §2a Contamination null

**What:** what pure contamination scores in this cohort, per statistic.

**How:** the same tests applied to the lineage-control genes; thresholds come from the p90 row.

```csv
index,log2fc_paired,frac_donors_up,retained_trim,retained_clean,top1_share,top1_share_excess,top5_share,r_myeloid,r_myeloid_excess,r_umi,r_umi_excess,r_dbl,or_topdecile,or_topdecile_excess,rho_burden,top_cluster_share
p50,0,0.394,0,0.757,1,0,1,0.197,0.186,0.0178,-0.00362,0.0189,1.01,-0.171,0.0197,0.0851
p90,0.0178,0.583,1.04,1.12,1,0,1,0.287,0.278,0.0649,0.015,0.0251,1.8,0.0799,0.338,0.193
p97.5,0.0374,0.629,1.19,1.27,1,0,1,0.499,0.481,0.0926,0.0175,0.0342,1.92,0.225,0.359,0.419
```

## §2a Calibration verdict

**What:** whether the differential-contamination mechanism operates here at all.

**How:** the controls must themselves be up in the malignant arm; if they are not, no gene is excluded.

```csv
key,value
null_calibrated,False
verdict,"18 lineage controls: median log2FC +0.00, median frac_donors_up 0.39 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first."
```

## §2b Artifact verdict, per gene

**What:** the six tests for every gene up at q < 0.05.

**How:** retained_* = share of the effect surviving trimming / winsorising / dropping dirty cells / depth matching; *_excess = prevalence-matched residual against the null; rho_burden = Spearman of the per-donor effect against that donor's malignant:reactive ratio. Two failures = artifact, one = suspect.

**Full table on disk:** `tables/tgt43_artifact_evidence.csv`

```csv
index,gene,log2fc_paired,frac_donors_up,median_pct_expr,retained_trim,retained_wins,retained_clean,retained_ds,top1_share_excess,r_myeloid_excess,r_umi_excess,or_topdecile_excess,rho_burden,top_cluster_share,n_fails,fail_reasons
0,KIR3DL2,2.79,0.904,0.367,1.05,1.25,1.07,1.03,-0.00776,-0.0109,-0.0435,0.0652,-0.145,0.00836,0,- (null not calibrated; no exclusion applied)
1,CD70,1.93,0.885,0.241,1.03,1.35,0.989,0.885,0.0371,0.0203,0.0707,0.42,-0.286,0.012,2,- (null not calibrated; no exclusion applied)
2,TPO,1.85,0.962,0.123,1.02,0.973,0.988,0.94,0.0148,0.000668,-0.0276,-0.376,0.0113,0.0111,1,- (null not calibrated; no exclusion applied)
3,TIGIT,1.65,0.827,0.438,1.03,1.43,1.07,1.01,0.026,-0.0104,-0.0191,-0.172,-0.199,0.00742,1,- (null not calibrated; no exclusion applied)
4,STAM,1.64,0.865,0.445,1.06,1.18,1.05,0.951,0.00506,-0.00188,-0.0202,-0.287,-0.122,0.00996,1,- (null not calibrated; no exclusion applied)
5,FCRL3,1.54,0.712,0.195,1.01,1.41,1.02,0.945,0.00011,-0.00875,-0.0412,-0.379,-0.146,0.0137,1,- (null not calibrated; no exclusion applied)
6,GPR15,1.44,0.827,0.18,1.09,1.47,1.07,0.945,0.0304,-0.0134,-0.0371,-0.26,-0.0667,0.0148,1,- (null not calibrated; no exclusion applied)
7,PDCD1,1.22,0.865,0.108,0.966,0.544,1.04,0.935,0.00261,-0.00428,-0.00131,-0.172,-0.297,0.017,1,- (null not calibrated; no exclusion applied)
8,NRROS,1.2,0.904,0.157,0.945,1.17,1.02,0.924,-0.0293,0.01,0.0189,0.104,-0.0752,0.0132,1,- (null not calibrated; no exclusion applied)
9,PI16,1.17,0.846,0.254,1.08,1.29,1.04,1.06,0.0227,0.0161,-0.0284,-0.375,-0.105,0.0133,1,- (null not calibrated; no exclusion applied)
10,ITGB1,1.16,0.904,0.828,1.02,1.02,1.02,0.994,-0.00168,-0.0147,-0.091,-0.161,-0.235,0.00589,0,- (null not calibrated; no exclusion applied)
11,ADGRE1,1.15,0.865,0.096,1,0,1.04,0.869,-0.0253,0.00116,0.0247,0.142,-0.0389,0.0121,1,- (null not calibrated; no exclusion applied)
12,FGFR1,1.15,0.885,0.0968,0.992,0,1.11,0.796,0.00284,0.00513,0.000903,-0.0123,-0.0822,0.0146,1,- (null not calibrated; no exclusion applied)
13,CCR4,1.13,0.769,0.21,1.03,1.19,1.02,1,0.00974,-0.00108,-0.0413,-0.0605,-0.307,0.013,1,- (null not calibrated; no exclusion applied)
14,CPM,1.1,0.904,0.0607,0.89,0,1.02,0.708,0.00385,-0.00873,-0.0309,-0.454,-0.24,0.0231,1,- (null not calibrated; no exclusion applied)
15,CD84,1.07,0.981,0.247,1.03,1.49,1.03,0.939,-0.0209,0,-0.00862,-0.0523,-0.00307,0.0149,0,- (null not calibrated; no exclusion applied)
16,EPHA4,0.989,0.942,0.202,1.09,2,1.04,1.06,0,-0.0102,0.00436,-0.084,-0.273,0.0132,1,- (null not calibrated; no exclusion applied)
17,PTH2R,0.971,0.865,0.0401,0.776,0,0.813,0.819,0.0506,0.0109,-0.0104,-0.148,-0.132,0.0525,1,- (null not calibrated; no exclusion applied)
18,GPR55,0.968,0.904,0.0693,0.928,0,0.993,0.824,0.028,-0.0103,-0.0399,-0.136,-0.0614,0.0172,1,- (null not calibrated; no exclusion applied)
19,TSPAN2,0.938,0.923,0.138,1.06,1.88,1.06,1.06,-0.00353,0.00975,0.0169,0.213,-0.31,0.0163,1,- (null not calibrated; no exclusion applied)
20,ABCC1,0.907,0.904,0.221,1.05,1.57,1.06,0.975,-0.0103,-0.0094,-0.0241,0.00511,-0.0425,0.0136,0,- (null not calibrated; no exclusion applied)
21,ACVR2A,0.886,0.923,0.0793,1.07,0,1.01,0.904,-0.00798,-0.0141,-0.0486,-0.266,0.0453,0.0202,0,- (null not calibrated; no exclusion applied)
22,MBP,0.843,0.942,0.727,1.03,1.05,1,1.16,0.000746,-0.0136,-0.0908,-0.463,-0.209,0.00632,1,- (null not calibrated; no exclusion applied)
23,LAG3,0.841,0.75,0.088,1.06,0,1.08,0.987,-0.00275,0.000461,-0.0114,-0.222,-0.116,0.0281,0,- (null not calibrated; no exclusion applied)
24,EVC2,0.841,0.962,0.0506,0.915,0,0.982,0.805,-0.0107,-0.00262,-0.0209,-0.218,-0.151,0.0247,0,- (null not calibrated; no exclusion applied)
25,GPR146,0.824,0.904,0.11,0.984,0,0.977,1.05,0.0283,-0.00541,-0.00403,-0.104,0.00495,0.0149,1,- (null not calibrated; no exclusion applied)
26,FUT8,0.793,0.827,0.227,1.04,1.3,1.15,0.9,-0.00302,0.00184,-0.0169,0.154,-0.116,0.015,1,- (null not calibrated; no exclusion applied)
27,SLAMF6,0.771,0.923,0.259,1.07,1.26,1.07,1.13,0.00523,-0.00772,-0.0433,-0.473,0.0577,0.00978,1,- (null not calibrated; no exclusion applied)
28,CD82,0.765,0.846,0.439,1.01,1.16,1.05,1.2,0.00276,0.0219,0.0395,-0.182,-0.156,0.00701,2,- (null not calibrated; no exclusion applied)
29,TFRC,0.746,0.962,0.126,1.07,1.69,1.08,0.879,0.00194,0.00555,-0.00647,0.0595,-0.0596,0.0185,1,- (null not calibrated; no exclusion applied)
30,IL6R,0.738,0.923,0.243,1.02,1.13,0.992,1.07,-0.0166,0.00197,0.00466,-0.0567,-0.113,0.00775,0,- (null not calibrated; no exclusion applied)
31,ANXA2,0.706,0.769,0.508,1.01,1.07,0.982,1.16,0.00545,0.0265,0.102,0.522,-0.26,0.00747,2,- (null not calibrated; no exclusion applied)
32,F5,0.684,0.692,0.0889,0.973,0,0.941,0.833,0.0195,-0.00815,-0.0198,-0.124,-0.0984,0.0137,1,- (null not calibrated; no exclusion applied)
33,TNFRSF1B,0.654,0.577,0.278,1,1.05,0.922,0.785,0.0129,0.0193,0.00413,0.212,-0.215,0.0108,2,- (null not calibrated; no exclusion applied)
34,SIRPG,0.633,0.865,0.225,1.02,1.04,1.08,1.12,-0.00546,0.0105,0.0355,0.159,0,0.00901,1,- (null not calibrated; no exclusion applied)
35,GDPD5,0.628,0.788,0.0804,0.974,0,0.932,0.753,-0.0173,0.000443,-0.0165,-0.153,-0.136,0.0148,0,- (null not calibrated; no exclusion applied)
36,GRN,0.624,0.827,0.155,0.982,1.04,1.04,1.08,-0.0261,0.026,0.042,0.283,-0.106,0.0107,1,- (null not calibrated; no exclusion applied)
37,CTLA4,0.623,0.731,0.163,1.08,1.98,1,0.977,0.0389,0.00506,-0.00675,-0.0574,-0.178,0.017,1,- (null not calibrated; no exclusion applied)
38,CD28,0.605,0.865,0.413,1.05,1.14,1,1.1,0.00632,-0.00746,-0.00246,-0.257,-0.347,0.00904,1,- (null not calibrated; no exclusion applied)
39,TNFRSF25,0.602,0.731,0.395,1.01,1.02,1.17,0.873,-0.0108,0.00397,-0.0236,0.13,-0.31,0.00731,1,- (null not calibrated; no exclusion applied)
40,ICAM2,0.58,0.75,0.526,1.03,1.06,1.01,1.02,-0.0122,-0.0165,-0.0303,-0.271,0.00862,0.00606,0,- (null not calibrated; no exclusion applied)
41,SLAMF1,0.578,0.788,0.213,0.992,1.35,0.891,1.26,-0.0151,0.01,-0.0141,0.00973,-0.285,0.00971,0,- (null not calibrated; no exclusion applied)
42,MFGE8,0.577,0.808,0.0938,1.06,0,0.97,0.895,-0.0247,-0.00482,0.0141,-0.0116,-0.114,0.00884,0,- (null not calibrated; no exclusion applied)
43,GPR155,0.576,0.827,0.173,1.07,2,1.63,1.25,0.0163,-0.00698,0.00619,0.0599,-0.0679,0.0106,1,- (null not calibrated; no exclusion applied)
44,PTGER4,0.556,0.808,0.283,1.02,1.14,1.09,1.19,0.00887,-0.00881,-0.048,-0.401,-0.0799,0.00933,1,- (null not calibrated; no exclusion applied)
45,SIT1,0.554,0.712,0.339,0.909,0.975,1.01,0.91,-0.00227,0.0131,0.0377,0.192,-0.125,0.0082,1,- (null not calibrated; no exclusion applied)
46,FGL2,0.55,0.769,0.0435,0.796,0,1.09,1.11,0.0568,-0.00428,-0.033,-0.419,-0.0655,0.0227,1,- (null not calibrated; no exclusion applied)
47,CD79B,0.548,0.788,0.125,1.06,1.6,1.02,0.955,-0.0011,0.0174,0.0203,0.0368,0.061,0.0104,1,- (null not calibrated; no exclusion applied)
48,IGSF8,0.536,0.962,0.18,1.01,1.41,1.1,1.12,-0.00609,0.00447,0.0266,0.0894,0.107,0.00907,1,- (null not calibrated; no exclusion applied)
49,SLC5A3,0.535,0.75,0.105,0.961,0,1.09,0.782,0.0271,-0.0103,-0.0384,-0.232,0.0577,0.0172,1,- (null not calibrated; no exclusion applied)
50,TNFRSF4,0.533,0.673,0.225,1.03,1.24,0.998,0.981,0.0513,0.0085,-0.00432,-0.054,-0.0521,0.0192,1,- (null not calibrated; no exclusion applied)
51,CD27,0.528,0.827,0.629,0.999,1.07,1.06,1.31,-0.0139,-0.00272,0.0456,0.233,-0.0451,0.00604,1,- (null not calibrated; no exclusion applied)
52,ITGB2,0.505,0.827,0.635,0.966,1.03,0.928,1.11,0.00422,0.0191,0.0396,-0.175,0.00606,0.00585,2,- (null not calibrated; no exclusion applied)
53,TGFBR3,0.501,0.692,0.122,1.03,0.482,1.15,0.692,0.0129,-0.00745,-0.047,-0.241,-0.197,0.0142,1,- (null not calibrated; no exclusion applied)
54,IFNGR2,0.497,0.75,0.166,0.963,0.73,1.04,0.813,0.00454,-0.0113,-0.00356,-0.0419,0.13,0.0164,1,- (null not calibrated; no exclusion applied)
55,IGHM,0.488,0.788,0.0243,0.568,0,0.952,0.716,-0.125,0.00593,0.0211,0.156,-0.224,0.0391,1,- (null not calibrated; no exclusion applied)
56,PIEZO1,0.48,0.808,0.103,0.991,0.124,1.01,0.874,0.0112,0.00297,-0.0161,0.0102,-0.194,0.0186,1,- (null not calibrated; no exclusion applied)
57,TMEM245,0.474,0.731,0.182,1.01,1.03,1.03,0.689,0.00996,-0.000744,-0.00984,-0.144,-0.0372,0.0156,1,- (null not calibrated; no exclusion applied)
58,TOM1,0.461,0.808,0.188,1.02,1.16,0.902,1.04,-0.0122,-0.00472,0.0145,-0.144,-0.0873,0.0085,0,- (null not calibrated; no exclusion applied)
59,SLC1A5,0.46,0.865,0.0713,0.915,0,1.03,0.846,0.00884,0.00878,0.0422,0.285,0.14,0.0189,2,- (null not calibrated; no exclusion applied)
60,FLNA,0.458,0.846,0.633,0.99,1.08,1.02,1.12,0.0132,0.0225,0.021,-0.466,-0.307,0.00627,2,- (null not calibrated; no exclusion applied)
61,FCGRT,0.451,0.692,0.131,0.872,0.208,1.01,0.96,-0.00443,0.00726,0.00131,-0.271,0.241,0.0131,0,- (null not calibrated; no exclusion applied)
62,TM2D1,0.45,0.923,0.231,1.02,1.06,0.884,0.968,-0.0242,0.00628,0.0355,0.0679,-0.138,0.00561,1,- (null not calibrated; no exclusion applied)
63,NRN1,0.447,0.769,0.0197,0.501,0,0.578,0.701,-0.178,-0.00254,0.019,-0.0153,0.0975,0.0184,1,- (null not calibrated; no exclusion applied)
64,CSF2RB,0.444,0.75,0.0243,0.377,0,1.06,0.989,-0.205,0.00546,-0.0175,-0.236,0.0765,0.0193,1,- (null not calibrated; no exclusion applied)
65,GPR171,0.442,0.635,0.162,1.06,0.859,0.951,1.08,0.0229,-0.00355,-0.0092,0.0488,0.241,0.0155,1,- (null not calibrated; no exclusion applied)
66,CYTH1,0.431,0.942,0.64,1.03,1.09,1.12,1.19,0.00185,-0.0177,-0.0887,-1.06,-0.192,0.00588,1,- (null not calibrated; no exclusion applied)
67,SLC7A5,0.418,0.712,0.103,0.884,0,1.01,1.05,0.0291,0.00242,-0.0223,-0.0433,0.000683,0.0245,1,- (null not calibrated; no exclusion applied)
68,PCSK7,0.414,0.75,0.3,1.05,1.05,1.04,1.01,0.00213,-0.0014,-0.0275,-0.142,-0.000512,0.00991,1,- (null not calibrated; no exclusion applied)
69,CLDND1,0.413,0.75,0.444,1,1.08,0.843,0.885,0.00556,-0.0198,-0.071,-0.131,-0.104,0.00818,1,- (null not calibrated; no exclusion applied)
70,IL21R,0.411,0.712,0.132,1.03,0.971,0.95,0.927,-0.00556,-0.000668,0.00433,0.0502,-0.0347,0.0119,0,- (null not calibrated; no exclusion applied)
71,SLC38A5,0.405,0.788,0.0672,0.903,0,1.11,0.884,0.000708,-0.00299,0.0355,0.217,0.0464,0.0184,2,- (null not calibrated; no exclusion applied)
72,ENTPD1,0.404,0.692,0.0456,0.62,0,1.5,0.848,-0.00769,0.0119,0.0128,0.141,-0.0844,0.0216,1,- (null not calibrated; no exclusion applied)
73,B3GNT2,0.404,0.788,0.163,1.06,1.33,1.13,1.3,0.00894,0.013,0.0227,0.0576,-0.186,0.011,2,- (null not calibrated; no exclusion applied)
74,TMX4,0.403,0.846,0.302,1.05,1.13,1.06,1.15,0,0.00832,0.00998,-0.159,-0.0739,0.00733,1,- (null not calibrated; no exclusion applied)
75,CCR7,0.401,0.712,0.64,1.02,1.03,1.2,1.04,0.0119,-0.0034,0.012,-0.681,-0.0307,0.00674,1,- (null not calibrated; no exclusion applied)
76,SYPL1,0.387,0.885,0.378,1.05,1.05,1,1.35,-0.0166,-0.000185,0.00983,0.0904,-0.248,0.00607,1,- (null not calibrated; no exclusion applied)
77,NT5E,0.378,0.615,0.0291,0.485,0,0.846,0.514,0.0329,0.000355,-0.00665,-0.0957,0.0889,0.0202,2,- (null not calibrated; no exclusion applied)
78,EMP3,0.377,0.846,0.926,1.01,1.02,0.995,1.27,-0.0114,0.00614,-0.0332,0.555,-0.155,0.00403,1,- (null not calibrated; no exclusion applied)
79,RAB25,0.375,0.769,0.025,0.936,0,0.796,0.849,-0.174,-0.00326,0.0341,-0.0894,0.147,0.0189,1,- (null not calibrated; no exclusion applied)
[181 further rows omitted from this block]
```

## §2c Three-arm detection

**What:** median detection of the disputed genes and the controls in healthy, reactive and malignant CD4.

**How:** median over donors of the share of cells with >= 1 UMI. Contamination has no reason to spare healthy CD4.

```csv
gene,CD4_HC,CD4_malignant,CD4_reactive
IL2RA,0.0747,0.0383,0.0636
PTH2R,0.000288,0.0401,0.000912
EVC2,0.00382,0.0506,0.00642
ADGRE1,0.0278,0.096,0.0237
FGFR1,0.0115,0.0968,0.0166
CD7,0.665,0.101,0.647
TPO,0.00137,0.123,0.0052
NRROS,0.0545,0.157,0.0478
CCR4,0.07,0.21,0.0989
KIR3DL2,0.00758,0.367,0.0105
MBP,0.446,0.727,0.449
```

## §3 Candidates passing every gate

**What:** the genes selectively up on malignant CD4 that survived §2.

**How:** clean, q < 0.05, paired log2FC > 0.5, detected in >= 50% of donors, not a loss marker or lineage control.

```csv
index,gene,log2fc_paired,frac_donors_up,frac_donors_positive,median_pct_expr,q,z_blocked,retained_trim,retained_clean,retained_ds,rho_burden,worst_T_arm,cd_name,stage
0,KIR3DL2,2.79,0.904,0.776,0.367,8.09e-08,6.7,1.05,1.07,1.03,-0.145,NK,CD158k,phase 2
1,CD70,1.93,0.885,0.776,0.241,8.93e-08,6.53,1.03,0.989,0.885,-0.286,CD8,CD70,phase 1
2,TPO,1.85,0.962,0.569,0.123,8.09e-08,7.76,1.02,0.988,0.94,0.0113,CD4_reactive,,
3,TIGIT,1.65,0.827,0.931,0.438,4.19e-07,6.46,1.03,1.07,1.01,-0.199,NK,,
4,STAM,1.64,0.865,0.862,0.445,1.49e-07,5.79,1.06,1.05,0.951,-0.122,CD4_reactive,,
5,FCRL3,1.54,0.712,0.603,0.195,3.69e-06,5.08,1.01,1.02,0.945,-0.146,NK,,
6,GPR15,1.44,0.827,0.534,0.18,9.64e-07,4.83,1.09,1.07,0.945,-0.0667,CD4_reactive,,
7,PDCD1,1.22,0.865,0.534,0.108,2.29e-06,6.63,0.966,1.04,0.935,-0.297,CD8,PD-1,phase 2
8,NRROS,1.2,0.904,0.638,0.157,9.87e-08,6.19,0.945,1.02,0.924,-0.0752,NK,,
9,PI16,1.17,0.846,0.69,0.254,1.44e-06,5.18,1.08,1.04,1.06,-0.105,CD4_reactive,,
10,ITGB1,1.16,0.904,0.983,0.828,2.37e-07,7.17,1.02,1.02,0.994,-0.235,NK,,
13,CCR4,1.13,0.769,0.724,0.21,5.6e-05,3.78,1.03,1.02,1,-0.307,CD4_reactive,CCR4,approved
15,CD84,1.07,0.981,0.948,0.247,8.09e-08,7.55,1.03,1.03,0.939,-0.00307,CD4_reactive,CD84 / SLAMF5,phase 1/2
16,EPHA4,0.989,0.942,0.862,0.202,8.09e-08,7.25,1.09,1.04,1.06,-0.273,CD4_reactive,,
19,TSPAN2,0.938,0.923,0.638,0.138,2.79e-07,6.29,1.06,1.06,1.06,-0.31,NK,,
20,ABCC1,0.907,0.904,0.914,0.221,2.11e-07,5.42,1.05,1.06,0.975,-0.0425,CD4_reactive,,
22,MBP,0.843,0.942,1,0.727,8.09e-08,6.95,1.03,1,1.16,-0.209,NK,,
25,GPR146,0.824,0.904,0.517,0.11,8.84e-07,5.51,0.984,0.977,1.05,0.00495,CD4_reactive,,
26,FUT8,0.793,0.827,0.879,0.227,1.57e-06,5.72,1.04,1.15,0.9,-0.116,CD4_reactive,,
27,SLAMF6,0.771,0.923,0.879,0.259,8.38e-08,6.03,1.07,1.07,1.13,0.0577,NK,,
28,CD82,0.765,0.846,0.931,0.439,4.69e-07,5.27,1.01,1.05,1.2,-0.156,CD4_reactive,,
29,TFRC,0.746,0.962,0.655,0.126,8.09e-08,6.14,1.07,1.08,0.879,-0.0596,CD4_reactive,,
30,IL6R,0.738,0.923,0.983,0.243,9.87e-08,6.1,1.02,0.992,1.07,-0.113,CD4_reactive,,
31,ANXA2,0.706,0.769,0.983,0.508,3.2e-05,4.35,1.01,0.982,1.16,-0.26,NK,,
33,TNFRSF1B,0.654,0.577,0.862,0.278,0.00653,1.62,1,0.922,0.785,-0.215,NK,TNFR2,phase 1/2a
34,SIRPG,0.633,0.865,0.828,0.225,1.67e-06,4.43,1.02,1.08,1.12,0,CD4_reactive,,
36,GRN,0.624,0.827,0.672,0.155,1.51e-06,5.15,0.982,1.04,1.08,-0.106,NK,,
37,CTLA4,0.623,0.731,0.741,0.163,0.000316,4.03,1.08,1,0.977,-0.178,CD4_reactive,,
38,CD28,0.605,0.865,1,0.413,2.87e-07,5.99,1.05,1,1.1,-0.347,CD4_reactive,,
39,TNFRSF25,0.602,0.731,1,0.395,0.000135,3.02,1.01,1.17,0.873,-0.31,CD4_reactive,,
40,ICAM2,0.58,0.75,1,0.526,6.54e-06,4.4,1.03,1.01,1.02,0.00862,CD4_reactive,,
41,SLAMF1,0.578,0.788,0.81,0.213,4.86e-05,3.57,0.992,0.891,1.26,-0.285,CD4_reactive,,
43,GPR155,0.576,0.827,0.759,0.173,4.83e-06,4.96,1.07,1.63,1.25,-0.0679,CD4_reactive,,
44,PTGER4,0.556,0.808,0.914,0.283,2.16e-05,3.34,1.02,1.09,1.19,-0.0799,NK,,
45,SIT1,0.554,0.712,0.931,0.339,5.23e-05,2.95,0.909,1.01,0.91,-0.125,CD4_reactive,,
47,CD79B,0.548,0.788,0.621,0.125,0.000161,4.46,1.06,1.02,0.955,0.061,NK,,
48,IGSF8,0.536,0.962,0.862,0.18,1.12e-07,5.47,1.01,1.1,1.12,0.107,NK,,
49,SLC5A3,0.535,0.75,0.534,0.105,9.3e-05,4.28,0.961,1.09,0.782,0.0577,NK,,
50,TNFRSF4,0.533,0.673,0.69,0.225,0.00451,2.44,1.03,0.998,0.981,-0.0521,CD4_reactive,,
52,ITGB2,0.505,0.827,1,0.635,2.57e-06,4.17,0.966,0.928,1.11,0.00606,NK,,
53,TGFBR3,0.501,0.692,0.586,0.122,0.0067,2.65,1.03,1.15,0.692,-0.197,NK,,
```

## §3 Loss markers

**What:** the Sezary loss phenotype, as the check that cleaning did not flatten real biology.

**How:** same paired test.

```csv
index,gene,log2fc_paired,frac_donors_up,q,artifact_flag
51,CD27,0.528,0.827,2.01e-06,clean
915,CD5,-0.0125,0.481,0.805,clean
1047,CD6,-0.459,0.25,9.64e-05,clean
1052,CD2,-0.504,0.192,1.11e-05,clean
1094,DPP4,-1.91,0.0192,8.09e-08,clean
1095,CD7,-1.99,0.115,1.76e-07,clean
```

## §3 FD Per-gene summary of the donor x gene matrix

**What:** how consistent each leading antigen is across patients.

**How:** per-donor log2FC; frac_up_1 is the share of donors where the antigen is at least doubled.

```csv
gene,n_donors,median,iqr,min,max,frac_up,frac_up_1
KIR3DL2,52,2.79,1.84,-0.471,4.59,0.904,0.769
CD70,52,1.93,1.07,-0.886,4.15,0.885,0.846
TPO,52,1.85,2.43,-0.0379,5.04,0.962,0.692
TIGIT,52,1.65,1.47,-1.29,4.34,0.827,0.731
STAM,52,1.64,1.01,-1.41,3.98,0.865,0.75
FCRL3,52,1.54,2.67,-1.1,4.43,0.712,0.577
GPR15,52,1.44,2.48,-1.08,4.5,0.827,0.596
PDCD1,52,1.22,1.01,-3.46,3.35,0.865,0.577
NRROS,52,1.2,0.982,-0.589,2.31,0.904,0.654
PI16,52,1.17,1.6,-2.65,3.15,0.846,0.596
ITGB1,52,1.16,0.897,-2.28,3.01,0.904,0.635
CCR4,52,1.13,1.51,-3.28,3.82,0.769,0.558
```

## §3 FE Eligibility

**What:** the fraction of patients in whom at least X% of malignant CD4 cells are positive.

**How:** transcript detection per donor, thresholded.

**Full table on disk:** `tables/tgt43_eligibility_blood.csv`

```csv
index,gene,n_donors,median_pct_expr,frac_donors_ge10,frac_donors_ge25,frac_donors_ge50,frac_donors_ge75
4,ITGB1,52,0.828,0.981,0.981,0.923,0.673
10,TIGIT,52,0.424,0.923,0.75,0.327,0.154
9,STAM,52,0.445,0.885,0.769,0.404,0.0192
1,CD70,52,0.241,0.808,0.481,0.0577,0
5,KIR3DL2,52,0.367,0.769,0.615,0.423,0.115
0,CCR4,52,0.231,0.731,0.462,0.115,0
8,PI16,52,0.254,0.712,0.5,0.25,0.0192
6,NRROS,52,0.149,0.615,0.308,0,0
2,FCRL3,52,0.198,0.596,0.385,0.115,0
3,GPR15,52,0.182,0.558,0.385,0.115,0.0192
11,TPO,52,0.121,0.558,0.365,0.173,0.0192
7,PDCD1,52,0.113,0.519,0.192,0.0192,0
```

## §4 Off-tumour, T and NK arms only

**What:** the worst non-target T arm and the healthy-CD4 burden for each candidate.

**How:** share of cells detected. Monocytes, DC, B, plasma, platelets and HSPC are NOT in this table.

```csv
index,gene,cd_name,worst_T_arm,worst_T_arm_pct,healthy_t_burden,median_pct_expr,frac_donors_positive
22,MBP,,NK,0.781,0.446,0.727,1
10,ITGB1,,NK,0.523,0.391,0.828,0.983
31,ANXA2,,NK,0.519,0.321,0.508,0.983
33,TNFRSF1B,TNFR2,NK,0.461,0.205,0.278,0.862
39,TNFRSF25,,CD4_reactive,0.304,0.204,0.395,1
3,TIGIT,,NK,0.283,0.123,0.438,0.931
38,CD28,,CD4_reactive,0.264,0.227,0.413,1
28,CD82,,CD4_reactive,0.234,0.216,0.439,0.931
34,SIRPG,,CD4_reactive,0.187,0.194,0.225,0.828
27,SLAMF6,,NK,0.176,0.11,0.259,0.879
30,IL6R,,CD4_reactive,0.171,0.153,0.243,0.983
36,GRN,,NK,0.163,0.0906,0.155,0.672
8,NRROS,,NK,0.152,0.0545,0.157,0.638
4,STAM,,CD4_reactive,0.122,0.103,0.445,0.862
5,FCRL3,,NK,0.121,0.0131,0.195,0.603
26,FUT8,,CD4_reactive,0.117,0.0862,0.227,0.879
20,ABCC1,,CD4_reactive,0.107,0.0699,0.221,0.914
15,CD84,CD84 / SLAMF5,CD4_reactive,0.101,0.0908,0.247,0.948
13,CCR4,CCR4,CD4_reactive,0.0989,0.07,0.21,0.724
37,CTLA4,,CD4_reactive,0.0916,0.0847,0.163,0.741
19,TSPAN2,,NK,0.0915,0.0346,0.138,0.638
0,KIR3DL2,CD158k,NK,0.0894,0.00758,0.367,0.776
9,PI16,,CD4_reactive,0.0766,0.0586,0.254,0.69
16,EPHA4,,CD4_reactive,0.0616,0.0493,0.202,0.862
29,TFRC,,CD4_reactive,0.0573,0.0387,0.126,0.655
25,GPR146,,CD4_reactive,0.0463,0.0462,0.11,0.517
7,PDCD1,PD-1,CD8,0.0408,0.0289,0.108,0.534
1,CD70,CD70,CD8,0.0353,0.0168,0.241,0.776
6,GPR15,,CD4_reactive,0.0203,0.0123,0.18,0.534
2,TPO,,CD4_reactive,0.0052,0.00137,0.123,0.569
```

## §5 Trafficking modules per arm

**What:** the state of the machinery that puts a protein on the surface.

**How:** per-module z across donors, then the arm mean. Context, not a correction.

```csv
module,CD4_HC,CD4_malignant,CD4_reactive,CD8,NK
endocytosis,-0.304,0.0231,-0.042,0.0219,0.521
er_folding_qc,-0.136,-0.0881,-0.375,0.248,1.71
er_to_golgi,0.0461,0.248,-0.444,0.0541,1.08
er_translocation,-0.0745,0.0917,-0.257,0.06,0.981
escrt_exosome,-0.398,1.03,-0.294,-0.491,-0.489
glycosylation,-0.403,0.73,-0.316,-0.351,0.965
recycling,-0.654,0.501,-0.337,-0.0757,0.569
retrograde,0.00831,-0.216,-0.161,0.241,0.71
shedding,-0.0813,0.0881,0.123,-0.218,0.751
snare_fusion,-0.393,-0.411,-0.399,0.58,1.59
ubiquitin_proteostasis,-0.619,0.487,-0.215,-0.147,0.338
```

## §6 Escape risk

**What:** how much each candidate varies between patients and within one patient's clusters.

**How:** SD of detection across donors, and across a donor's own clusters.

```csv
index,gene,between_donor_sd,min_donor_pct,within_donor_sd
4,ABCC1,0.143,0.0311,0.0829
50,ANXA2,0.228,0.0638,0.189
111,CCR4,0.186,0.00256,0.0995
144,CD28,0.153,0.133,0.126
168,CD70,0.168,0,0.107
174,CD82,0.197,0.0163,0.135
176,CD84,0.153,0.0865,0.104
265,CTLA4,0.172,0.00778,0.0723
326,EPHA4,0.12,0.0395,0.0865
362,FCRL3,0.194,0,0.0797
386,FUT8,0.136,0.0602,0.092
429,GPR146,0.117,0.00163,0.0536
430,GPR15,0.202,0.000355,0.0712
457,GRN,0.0985,0.0206,0.0663
536,IL6R,0.162,0.0891,0.0917
552,ITGB1,0.193,0.0455,0.121
570,KIR3DL2,0.288,0,0.156
643,MBP,0.173,0.269,0.134
703,NRROS,0.141,0.0112,0.0811
761,PDCD1,0.128,0.00198,0.0589
773,PI16,0.238,0,0.118
899,SIRPG,0.178,0.000261,0.116
902,SLAMF6,0.143,0.0506,0.0999
982,STAM,0.209,0.038,0.118
999,TFRC,0.0824,0.0225,0.0561
1009,TIGIT,0.226,0.0298,0.139
1038,TNFRSF1B,0.252,0.0241,0.101
1040,TNFRSF25,0.182,0.109,0.103
1048,TPO,0.241,0,0.103
1064,TSPAN2,0.18,0.000542,0.0906
```

## §7 Control check

**What:** where the antigens with a known answer sit in the ranking.

**How:** rank by paired log2FC over all ranked genes. TNFRSF8 is not testable: it is absent from the HVG object.

```csv
index,gene,rank,log2fc_paired,frac_donors_positive
0,KIR3DL2,1,2.79,0.776
13,CCR4,14,1.13,0.724
1002,IL2RA,1003,-0.19,0.259
```

## §7 Shortlist with the plausibility layer

**What:** the final candidate table, with surfaceome evidence beside the statistics.

**How:** surface_confident / n_tm / surfy_class come from CSPA + Surfy, not from this cohort.

**Full table on disk:** `tables/tgt43_shortlist_blood.csv`

```csv
index,gene,cd_name,frac_donors_positive,median_pct_expr,log2fc_paired,q,worst_T_arm,healthy_t_burden,between_donor_sd,stage,n_evidence,surface_confident,n_tm,surfy_class,artifact_flag
0,KIR3DL2,CD158k,0.776,0.367,2.79,8.09e-08,NK,0.00758,0.288,phase 2,1,True,1.0,Receptors,clean
1,CD70,CD70,0.776,0.241,1.93,8.93e-08,CD8,0.0168,0.168,phase 1,2,True,1.0,Unclassified,clean
2,TPO,,0.569,0.123,1.85,8.09e-08,CD4_reactive,0.00137,0.241,,2,True,1.0,Enzymes,clean
3,TIGIT,,0.931,0.438,1.65,4.19e-07,NK,0.123,0.226,,3,True,1.0,Miscellaneous,clean
4,STAM,,0.862,0.445,1.64,1.49e-07,CD4_reactive,0.103,0.209,,,,,,clean
5,FCRL3,,0.603,0.195,1.54,3.69e-06,NK,0.0131,0.194,,3,True,1.0,Receptors,clean
6,GPR15,,0.534,0.18,1.44,9.64e-07,CD4_reactive,0.0123,0.202,,1,True,7.0,Receptors,clean
7,PDCD1,PD-1,0.534,0.108,1.22,2.29e-06,CD8,0.0289,0.128,phase 2,3,True,1.0,Unclassified,clean
8,NRROS,,0.638,0.157,1.2,9.87e-08,NK,0.0545,0.141,,2,True,1.0,Miscellaneous,clean
9,PI16,,0.69,0.254,1.17,1.44e-06,CD4_reactive,0.0586,0.238,,2,True,1.0,Unclassified,clean
10,ITGB1,,0.983,0.828,1.16,2.37e-07,NK,0.391,0.193,,4,True,1.0,Receptors,clean
13,CCR4,CCR4,0.724,0.21,1.13,5.6e-05,CD4_reactive,0.07,0.186,approved,4,True,7.0,Receptors,clean
15,CD84,CD84 / SLAMF5,0.948,0.247,1.07,8.09e-08,CD4_reactive,0.0908,0.153,phase 1/2,4,True,1.0,Unclassified,clean
16,EPHA4,,0.862,0.202,0.989,8.09e-08,CD4_reactive,0.0493,0.12,,3,True,1.0,Receptors,clean
19,TSPAN2,,0.638,0.138,0.938,2.79e-07,NK,0.0346,0.18,,2,True,4.0,Miscellaneous,clean
20,ABCC1,,0.914,0.221,0.907,2.11e-07,CD4_reactive,0.0699,0.143,,3,True,17.0,Transporters,clean
22,MBP,,1,0.727,0.843,8.09e-08,NK,0.446,0.173,,2,True,,,clean
25,GPR146,,0.517,0.11,0.824,8.84e-07,CD4_reactive,0.0462,0.117,,1,True,7.0,Receptors,clean
26,FUT8,,0.879,0.227,0.793,1.57e-06,CD4_reactive,0.0862,0.136,,,,,,clean
27,SLAMF6,,0.879,0.259,0.771,8.38e-08,NK,0.11,0.143,,4,True,1.0,Unclassified,clean
28,CD82,,0.931,0.439,0.765,4.69e-07,CD4_reactive,0.216,0.197,,2,True,4.0,Miscellaneous,clean
29,TFRC,,0.655,0.126,0.746,8.09e-08,CD4_reactive,0.0387,0.0824,,2,True,,,clean
30,IL6R,,0.983,0.243,0.738,9.87e-08,CD4_reactive,0.153,0.162,,2,True,1.0,Receptors,clean
31,ANXA2,,0.983,0.508,0.706,3.2e-05,NK,0.321,0.228,,2,True,,,clean
33,TNFRSF1B,TNFR2,0.862,0.278,0.654,0.00653,NK,0.205,0.252,phase 1/2a,1,True,1.0,Receptors,clean
34,SIRPG,,0.828,0.225,0.633,1.67e-06,CD4_reactive,0.194,0.178,,2,True,1.0,Miscellaneous,clean
36,GRN,,0.672,0.155,0.624,1.51e-06,NK,0.0906,0.0985,,1,True,,,clean
37,CTLA4,,0.741,0.163,0.623,0.000316,CD4_reactive,0.0847,0.172,,2,True,1.0,Unclassified,clean
38,CD28,,1,0.413,0.605,2.87e-07,CD4_reactive,0.227,0.153,,4,True,1.0,Unclassified,clean
39,TNFRSF25,,1,0.395,0.602,0.000135,CD4_reactive,0.204,0.182,,3,True,1.0,Receptors,clean
```

## §2d TCR-call control

**What:** the same screen with the arms built from the expanded TCR clone instead of the Leiden cluster.

**How:** an effect that is an artifact of how the clusters were drawn cannot survive a call made from the TCR.

**Full table on disk:** `tables/tgt43_call_sensitivity.csv`

```csv
index,gene,log2fc_paired_cluster,frac_donors_up_cluster,n_donors_paired_cluster,log2fc_paired_tcr,frac_donors_up_tcr,n_donors_paired_tcr
0,KIR3DL2,2.79,0.904,52,2.06,0.906,53
2,TPO,1.85,0.962,52,1.22,0.925,53
8,NRROS,1.2,0.904,52,0.811,0.887,53
11,ADGRE1,1.15,0.865,52,0.801,0.887,53
12,FGFR1,1.15,0.885,52,0.766,0.849,53
13,CCR4,1.13,0.769,52,0.803,0.717,53
17,PTH2R,0.971,0.865,52,0.583,0.887,53
22,MBP,0.843,0.942,52,0.673,0.925,53
24,EVC2,0.841,0.962,52,0.57,0.887,53
51,CD27,0.528,0.827,52,0.588,0.868,53
915,CD5,-0.0125,0.481,52,0.0553,0.509,53
1002,IL2RA,-0.19,0.385,52,-0.259,0.283,53
1047,CD6,-0.459,0.25,52,-0.329,0.264,53
1052,CD2,-0.504,0.192,52,-0.398,0.17,53
1094,DPP4,-1.91,0.0192,52,-1.69,0.0189,53
1095,CD7,-1.99,0.115,52,-1.84,0.151,53
```

## §8 Selective in skin AND blood

**What:** antigens that separate tumour from reactive CD4 in both compartments.

**How:** q < 0.05 and log2FC > 0.5 on both sides, after both screens were cohort-fixed and artifact-tested.

**Full table on disk:** `tables/tgt43_shared_skin_blood.csv`

```csv
index,gene,cd_name_skin,log2fc_paired_skin,log2fc_paired_blood,frac_donors_positive_skin,frac_donors_positive_blood,stage_skin
0,CD70,CD70,1.1,1.93,0.808,0.776,phase 1
2,CPM,,1.02,1.1,0.577,0.293,
5,ADGRE1,,0.792,1.15,0.308,0.483,
6,GRN,,0.778,0.624,0.846,0.672,
8,TFRC,,0.731,0.746,0.949,0.655,
9,TNFRSF25,,0.698,0.602,0.872,1,
13,FUT8,,0.635,0.793,0.744,0.879,
14,CD82,,0.632,0.765,1,0.931,
15,EVC2,,0.624,0.841,0.231,0.155,
16,SLC5A3,,0.617,0.535,0.769,0.534,
17,IL6R,,0.614,0.738,0.769,0.983,
22,TNFRSF4,,0.542,0.533,0.897,0.69,
23,TSPAN2,,0.53,0.938,0.654,0.638,
24,PTH2R,,0.53,0.971,0.308,0.293,
```

======================== COPY TO HERE ==========================