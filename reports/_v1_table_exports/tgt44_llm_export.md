> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 44 — Blood stratification and heterogeneity

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**. This notebook stratifies the malignant-vs-reactive CD4 antigen
axis that notebook 43 established, and measures how much it varies between patients. It derives
no new screen: it reads 43's cleaned cohort and tables.

Cohort: 52 donors (52 people, 10 studies)
carrying BOTH a malignant and a reactive CD4 arm of >= 20 cells -- 37 Sezary, 8 MF, 5 unresolved,
2 MF/SS. The geskin2026 ex vivo drug wells and duplicate persons were removed in 43 section 0b,
and five healthy donors the atlas labelled SS were moved out of the reactive arm.

Per-donor effect: for each gene and donor, log2((mean malignant + 0.05)/(mean reactive + 0.05))
on log1p CP10K. Everything below is a function of that donor x gene matrix.

Stratified tests: the paired Wilcoxon is rerun inside each stratum, and `p_interaction` compares
the per-donor effects between strata (BH as `q_interaction`). Per-gene OLS regresses the donor
effect on diagnosis, ECP, sex, age and burden at once.

Heterogeneity: DerSimonian-Laird random effects (`mu` pooled effect, `tau` between-patient SD,
`I2`, `pi_lo`/`pi_hi` prediction interval, `q_Q`); COPA and the outlier sum with a
1000-draw sign-flip null; two-component Gaussian mixture BIC; hierarchical donor clustering
with a silhouette cut and at least 3 donors per cluster.

Power limits that apply to every result: treatment is almost entirely confounded with study (ECP
is the only agent with two usable arms, and it is concentrated in buus and ren); there are 2
early-stage donors against 35 advanced; response and survival are essentially absent. I2
saturates near 1 because each donor contributes thousands of cells, so rank consistency on `tau`,
not on I2. Everything is transcript, in a 10,000-gene HVG subset.

## §1 Power table

**What:** coverage of every candidate stratification over the paired donors.

**How:** count of non-missing values per field, split by diagnosis.

**Full table on disk:** `tables/tgt44_power_table.csv`

```csv
field,n,MF,SS,levels
dx2 (MF vs SS),52,8,37,"SS:37, MF:8, unresolved:7"
sex,52,8,37,"M:30, F:22"
age,44,7,31,"median 70, range 32-85"
stage,36,6,24,"IVA1:13, IVA:9, IVB:4, IIB:3, IVA2:3"
stage class,36,6,24,"advanced:34, early:2"
B stage,25,6,17,"B2:13, B1:11, B0:1"
blood tumour %,20,5,11,"median 59, range 13-98"
treatment at sampling known,44,8,34,"median 1, range 0-1"
treatment-naive,44,8,31,"no:40, yes:4"
prior lines,4,0,0,"median 2, range 0-4"
response,3,1,0,"mixed:2, PD:1"
outcome,4,1,1,"DOD:2, died_unspecified:1, died_other:1"
drug sensitivity,4,0,0,"romidepsin_resistant:3, romidepsin_sensitive:1"
malignant burden,52,8,37,"median 0.858, range 0.05-0.994"
```

## §1 Treatment at sampling

**What:** donors on each treatment class, and in which studies.

**How:** membership of the curated clin_current_tx_classes string; a donor can be on several.

**Full table on disk:** `tables/tgt44_treatment_counts.csv`

```csv
index,donors,MF,SS,studies
ecp,25,5,20,"borcherding2019, borcherding2023, buus2025, herrera2021, ren2023"
retinoid,14,4,10,"borcherding2023, buus2025, ren2023"
interferon,6,1,5,"buus2025, ren2023"
hdaci,6,1,4,"borcherding2019, borcherding2023, buus2025, geskin2026, ren2023"
systemic_steroid,2,0,2,buus2025
mogamulizumab,1,0,1,buus2025
methotrexate,1,0,1,buus2025
other_targeted,1,0,1,gaydosik2022
alemtuzumab,1,0,1,buus2025
```

## §2a Disease, MF vs SS, on 43's shortlist

**What:** each leading antigen's effect inside MF and inside SS, and whether they differ.

**How:** paired Wilcoxon within stratum; delta_lfc and q_interaction compare the per-donor effects. 8 MF donors: a consistency check, not a finding.

**Full table on disk:** `tables/tgt44_strat_disease.csv`

```csv
index,gene,frac_donors_up_MF,frac_donors_up_SS,log2fc_paired_MF,log2fc_paired_SS,n_donors_paired_MF,n_donors_paired_SS,delta_lfc,p_interaction,q_interaction,log2fc_all,q_all,on_shortlist
570,KIR3DL2,0.875,0.892,3.44,2.64,8,37,-0.794,0.61,1,2.79,8.09e-08,True
168,CD70,0.75,0.892,1.83,1.93,8,37,0.0976,0.738,1,1.93,8.93e-08,True
1048,TPO,0.875,0.973,1.19,1.91,8,37,0.716,0.298,0.992,1.85,8.09e-08,True
1009,TIGIT,0.75,0.838,1.67,1.83,8,37,0.162,0.988,1,1.65,4.19e-07,True
982,STAM,0.75,0.865,1.27,1.69,8,37,0.423,0.223,0.992,1.64,1.49e-07,True
362,FCRL3,0.75,0.703,1.66,1.43,8,37,-0.236,1,1,1.54,3.69e-06,True
430,GPR15,0.75,0.811,0.814,1.77,8,37,0.96,0.61,1,1.44,9.64e-07,True
761,PDCD1,0.875,0.865,1.28,1.14,8,37,-0.146,0.965,1,1.22,2.29e-06,True
703,NRROS,0.625,0.973,1,1.18,8,37,0.179,0.298,0.992,1.2,9.87e-08,True
773,PI16,0.625,0.892,0.898,1.11,8,37,0.211,0.285,0.992,1.17,1.44e-06,True
552,ITGB1,0.875,0.946,1.08,1.16,8,37,0.0835,0.805,1,1.16,2.37e-07,True
111,CCR4,0.75,0.811,1.63,1.1,8,37,-0.536,0.298,0.992,1.13,5.6e-05,True
176,CD84,1,1,0.91,0.843,8,37,-0.0671,0.827,1,1.07,8.09e-08,True
326,EPHA4,1,0.919,0.89,1,8,37,0.115,0.53,1,0.989,8.09e-08,True
1064,TSPAN2,1,0.946,0.938,1.02,8,37,0.0815,0.631,1,0.938,2.79e-07,True
4,ABCC1,1,0.865,0.77,1.01,8,37,0.242,0.422,1,0.907,2.11e-07,True
643,MBP,1,0.919,0.685,0.908,8,37,0.222,0.456,1,0.843,8.09e-08,True
429,GPR146,0.875,0.892,0.489,0.855,8,37,0.366,0.088,0.975,0.824,8.84e-07,True
386,FUT8,0.875,0.811,0.788,0.753,8,37,-0.0358,0.805,1,0.793,1.57e-06,True
902,SLAMF6,0.75,0.946,0.629,0.811,8,37,0.182,0.405,1,0.771,8.38e-08,True
174,CD82,0.75,0.865,1.02,0.684,8,37,-0.332,0.673,1,0.765,4.69e-07,True
999,TFRC,1,0.946,0.546,0.79,8,37,0.244,0.0824,0.975,0.746,8.09e-08,True
536,IL6R,0.75,0.946,0.421,0.785,8,37,0.364,0.088,0.975,0.738,9.87e-08,True
50,ANXA2,0.875,0.757,0.722,0.727,8,37,0.00511,0.896,1,0.706,3.2e-05,True
1038,TNFRSF1B,0.75,0.514,0.821,0.344,8,37,-0.476,0.357,0.995,0.654,0.00653,True
899,SIRPG,0.75,0.865,0.695,0.588,8,37,-0.108,0.827,1,0.633,1.67e-06,True
457,GRN,0.875,0.784,0.738,0.554,8,37,-0.185,0.474,1,0.624,1.51e-06,True
265,CTLA4,0.75,0.676,1.01,0.404,8,37,-0.609,0.357,0.995,0.623,0.000316,True
144,CD28,1,0.811,0.773,0.607,8,37,-0.166,0.439,1,0.605,2.87e-07,True
1040,TNFRSF25,0.75,0.73,0.532,0.586,8,37,0.0535,0.827,1,0.602,0.000135,True
```

## §2b ECP, on 43's shortlist

**What:** the same split by extracorporeal photopheresis at sampling.

**How:** as above; the study-blocked (van Elteren) version is the one to quote and is in the file named below.

**Full table on disk:** `tables/tgt44_strat_ecp_blocked.csv`

```csv
index,gene,log2fc_paired_ECP,log2fc_paired_no_ECP,n_donors_paired_ECP,n_donors_paired_no_ECP,delta_lfc,p_interaction,q_interaction,on_shortlist
4,ABCC1,0.946,1.01,25,19,-0.0657,0.538,0.946,True
50,ANXA2,0.756,0.711,25,19,0.0447,0.722,0.961,True
111,CCR4,1.19,1.25,25,19,-0.0693,0.586,0.949,True
144,CD28,0.706,0.607,25,19,0.0989,0.85,0.987,True
168,CD70,1.84,2.06,25,19,-0.216,0.67,0.956,True
174,CD82,0.811,0.626,25,19,0.185,0.636,0.949,True
176,CD84,1.05,0.84,25,19,0.21,0.619,0.949,True
265,CTLA4,0.733,0.587,25,19,0.146,0.687,0.957,True
326,EPHA4,0.975,1,25,19,-0.0281,0.943,1,True
362,FCRL3,1.88,0.976,25,19,0.903,0.169,0.788,True
386,FUT8,0.629,0.968,25,19,-0.339,0.102,0.778,True
429,GPR146,0.791,0.794,25,19,-0.00297,0.943,1,True
430,GPR15,2.26,1.08,25,19,1.18,0.209,0.841,True
457,GRN,0.627,0.651,25,19,-0.0235,0.868,0.988,True
536,IL6R,0.488,0.785,25,19,-0.297,0.236,0.861,True
552,ITGB1,1.16,1.16,25,19,0.000303,1,1,True
570,KIR3DL2,2.85,2.62,25,19,0.233,0.924,0.999,True
643,MBP,0.802,0.877,25,19,-0.0758,0.653,0.952,True
703,NRROS,1.08,1.25,25,19,-0.179,0.343,0.895,True
761,PDCD1,1.13,1.42,25,19,-0.287,0.246,0.865,True
773,PI16,1.09,1.25,25,19,-0.159,0.722,0.961,True
899,SIRPG,0.739,0.369,25,19,0.37,0.124,0.788,True
902,SLAMF6,0.933,0.56,25,19,0.374,0.0466,0.713,True
982,STAM,1.61,1.52,25,19,0.084,0.554,0.947,True
999,TFRC,0.79,0.762,25,19,0.0279,0.67,0.956,True
1009,TIGIT,1.91,1.41,25,19,0.492,0.492,0.938,True
1038,TNFRSF1B,0.847,-0.207,25,19,1.05,0.058,0.713,True
1040,TNFRSF25,0.745,0.554,25,19,0.191,0.463,0.922,True
1048,TPO,2.67,1.22,25,19,1.46,0.236,0.861,True
1064,TSPAN2,0.962,1.31,25,19,-0.349,0.619,0.949,True
```

## §2c Donor-level regression

**What:** which donor properties move each gene's effect, all terms in one model.

**How:** OLS per gene over donors, terms is_MF, ecp, is_F, age_z, burden_z; BH per term. Study is deliberately out of the design and is reported as its own ANOVA.

**Full table on disk:** `tables/tgt44_strat_regression.csv`

```csv
key,genes_at_q_lt_0.05
is_MF,0
ecp,0
is_F,0
age_z,0
burden_z,0
```

## §3 Tier 2 descriptive splits

**What:** effect sizes on each side of the splits too small to test.

**How:** median per-donor effect per side with a 400-draw bootstrap CI. No p-value.

**Full table on disk:** `tables/tgt44_tier2_descriptive.csv`

```csv
split,gene,n_yes,n_no,median_yes,median_no,delta,ci_lo,ci_hi
HDACi at sampling,KIR3DL2,6,38,4.13,2.69,1.45,-2.91,1.88
HDACi at sampling,CD70,6,38,2.49,1.78,0.709,-0.0265,2.03
HDACi at sampling,TPO,6,38,2.06,1.88,0.185,-1.7,2.28
HDACi at sampling,TIGIT,6,38,2.15,1.75,0.397,-0.544,1.04
HDACi at sampling,STAM,6,38,1.71,1.6,0.115,-0.693,0.572
HDACi at sampling,FCRL3,6,38,2.1,1.54,0.56,-1.02,2.29
HDACi at sampling,GPR15,6,38,2.74,1.44,1.3,0.168,2.8
HDACi at sampling,PDCD1,6,38,1.18,1.24,-0.062,-0.543,1.18
HDACi at sampling,NRROS,6,38,0.86,1.23,-0.367,-0.874,0.592
HDACi at sampling,PI16,6,38,1.26,1.2,0.0602,-1.02,0.658
HDACi at sampling,ITGB1,6,38,0.984,1.16,-0.177,-0.923,0.606
HDACi at sampling,CCR4,6,38,1.82,1.11,0.704,0.0434,1.69
HDACi at sampling,CD84,6,38,1.24,0.842,0.399,-0.337,0.618
HDACi at sampling,EPHA4,6,38,0.763,1,-0.241,-0.767,0.577
HDACi at sampling,TSPAN2,6,38,1.36,1.09,0.27,-1.23,1.33
interferon at sampling,KIR3DL2,6,38,2.84,2.79,0.049,-2.59,0.664
interferon at sampling,CD70,6,38,1.23,2,-0.773,-1.58,0.124
interferon at sampling,TPO,6,38,1.72,1.88,-0.161,-1.71,1.89
interferon at sampling,TIGIT,6,38,2.05,1.75,0.303,-1.53,0.852
interferon at sampling,STAM,6,38,1.86,1.57,0.294,-0.617,0.605
interferon at sampling,FCRL3,6,38,0.882,1.76,-0.875,-2.02,1.22
interferon at sampling,GPR15,6,38,0.519,1.97,-1.45,-3.18,0.396
interferon at sampling,PDCD1,6,38,1.48,1.18,0.303,-0.866,1.37
interferon at sampling,NRROS,6,38,1.16,1.2,-0.0423,-0.637,0.266
interferon at sampling,PI16,6,38,1.9,1.17,0.729,-1.28,1.61
interferon at sampling,ITGB1,6,38,1.51,1.14,0.369,-0.0841,0.803
interferon at sampling,CCR4,6,38,1.49,1.16,0.335,-0.73,0.88
interferon at sampling,CD84,6,38,1.41,0.828,0.583,-0.0773,1.53
interferon at sampling,EPHA4,6,38,1.42,0.914,0.509,0.0563,1.23
interferon at sampling,TSPAN2,6,38,1.22,1.06,0.155,-1.62,1.1
retinoid at sampling,KIR3DL2,14,30,2.7,2.91,-0.203,-1.93,0.743
retinoid at sampling,CD70,14,30,1.79,1.94,-0.153,-1.05,0.586
retinoid at sampling,TPO,14,30,3.08,1.53,1.56,-0.00938,2.36
retinoid at sampling,TIGIT,14,30,1.83,1.79,0.0426,-0.911,0.683
retinoid at sampling,STAM,14,30,1.72,1.6,0.117,-0.722,0.547
retinoid at sampling,FCRL3,14,30,1.9,1.44,0.465,-1.22,1.4
retinoid at sampling,GPR15,14,30,1.4,1.84,-0.438,-1.78,1.14
retinoid at sampling,PDCD1,14,30,0.99,1.24,-0.249,-1.22,0.259
retinoid at sampling,NRROS,14,30,0.976,1.26,-0.285,-0.766,0.0498
retinoid at sampling,PI16,14,30,1.41,1.17,0.235,-0.41,1.46
retinoid at sampling,ITGB1,14,30,1.15,1.17,-0.0186,-0.423,0.519
retinoid at sampling,CCR4,14,30,1.5,1.16,0.341,-0.605,0.814
retinoid at sampling,CD84,14,30,1.07,0.842,0.225,-0.616,0.419
retinoid at sampling,EPHA4,14,30,1.12,0.943,0.176,-0.243,0.634
retinoid at sampling,TSPAN2,14,30,0.991,1.22,-0.227,-0.979,0.719
ever on ECP,KIR3DL2,29,23,2.85,2.64,0.211,-1.03,1.04
ever on ECP,CD70,29,23,1.84,1.95,-0.113,-0.546,0.515
ever on ECP,TPO,29,23,2.67,1.3,1.37,0.214,2.08
ever on ECP,TIGIT,29,23,1.91,1.41,0.498,-0.275,1.32
ever on ECP,STAM,29,23,1.82,1.52,0.294,-0.261,0.647
ever on ECP,FCRL3,29,23,1.88,0.878,1,-0.356,2.13
ever on ECP,GPR15,29,23,2.17,1.06,1.11,-0.286,2.23
ever on ECP,PDCD1,29,23,1.14,1.26,-0.125,-0.721,0.512
ever on ECP,NRROS,29,23,1.08,1.3,-0.22,-0.825,0.208
ever on ECP,PI16,29,23,1.09,1.25,-0.159,-1.15,0.797
ever on ECP,ITGB1,29,23,1.15,1.16,-0.00705,-0.523,0.364
ever on ECP,CCR4,29,23,1.19,1.1,0.0883,-0.678,1
ever on ECP,CD84,29,23,1.05,1.33,-0.279,-0.619,0.493
ever on ECP,EPHA4,29,23,0.975,1,-0.0281,-0.581,0.344
ever on ECP,TSPAN2,29,23,0.869,1.21,-0.337,-1.28,0.385
[45 further rows omitted from this block]
```

## §4a Random-effects meta-analysis, 43's shortlist

**What:** pooled effect and between-patient spread for each leading antigen.

**How:** DerSimonian-Laird; `pi_lo > 0` is what 'works in every patient' means.

**Full table on disk:** `tables/tgt44_heterogeneity_meta.csv`

```csv
index,gene,n_donors,mu,se,tau2,tau,I2,Q,tau_over_mu,p_Q,pi_lo,pi_hi,frac_donors_up,se_source,q_Q,on_shortlist
643,MBP,52,0.793,0.0472,0.103,0.321,0.965,1.45e+03,0.405,2.3e-270,0.141,1.45,0.942,delta-method from per-arm cell SD,2.66e-270,True
144,CD28,52,0.578,0.0648,0.189,0.435,0.96,1.29e+03,0.751,7.58e-236,-0.304,1.46,0.865,delta-method from per-arm cell SD,8.65e-236,True
4,ABCC1,52,0.893,0.0716,0.221,0.471,0.928,707,0.527,3.46e-116,-0.0633,1.85,0.904,delta-method from per-arm cell SD,3.74e-116,True
457,GRN,52,0.637,0.0754,0.247,0.497,0.931,739,0.781,9.04e-123,-0.373,1.65,0.827,delta-method from per-arm cell SD,9.81e-123,True
899,SIRPG,52,0.584,0.0791,0.285,0.534,0.974,1.99e+03,0.914,0,-0.5,1.67,0.865,delta-method from per-arm cell SD,0,True
902,SLAMF6,52,0.827,0.0855,0.332,0.576,0.959,1.25e+03,0.696,4.72e-228,-0.342,2,0.923,delta-method from per-arm cell SD,5.37e-228,True
386,FUT8,52,0.712,0.0869,0.35,0.592,0.955,1.13e+03,0.831,6.36e-203,-0.49,1.91,0.827,delta-method from per-arm cell SD,7.2e-203,True
999,TFRC,52,0.814,0.0877,0.351,0.592,0.949,991,0.728,2.79e-174,-0.389,2.02,0.962,delta-method from per-arm cell SD,3.11e-174,True
536,IL6R,52,0.754,0.0892,0.366,0.605,0.972,1.83e+03,0.802,0,-0.474,1.98,0.923,delta-method from per-arm cell SD,0,True
174,CD82,52,0.763,0.0962,0.441,0.664,0.982,2.82e+03,0.87,0,-0.585,2.11,0.846,delta-method from per-arm cell SD,0,True
429,GPR146,52,0.862,0.099,0.46,0.679,0.966,1.52e+03,0.788,6.59e-285,-0.516,2.24,0.904,delta-method from per-arm cell SD,7.69e-285,True
1040,TNFRSF25,52,0.422,0.0994,0.484,0.696,0.984,3.17e+03,1.65,0,-0.99,1.83,0.731,delta-method from per-arm cell SD,0,True
50,ANXA2,52,0.648,0.101,0.505,0.71,0.99,4.87e+03,1.1,0,-0.793,2.09,0.769,delta-method from per-arm cell SD,0,True
176,CD84,52,1.13,0.109,0.564,0.751,0.976,2.1e+03,0.665,0,-0.394,2.65,0.981,delta-method from per-arm cell SD,0,True
703,NRROS,52,1.11,0.118,0.677,0.823,0.976,2.13e+03,0.742,0,-0.561,2.78,0.904,delta-method from per-arm cell SD,0,True
982,STAM,52,1.45,0.12,0.699,0.836,0.979,2.43e+03,0.577,0,-0.249,3.14,0.865,delta-method from per-arm cell SD,0,True
1064,TSPAN2,52,1.18,0.123,0.738,0.859,0.978,2.27e+03,0.73,0,-0.567,2.92,0.923,delta-method from per-arm cell SD,0,True
552,ITGB1,52,1.08,0.121,0.738,0.859,0.995,9.91e+03,0.795,0,-0.663,2.82,0.904,delta-method from per-arm cell SD,0,True
326,EPHA4,52,1.08,0.13,0.819,0.905,0.977,2.18e+03,0.837,0,-0.756,2.92,0.942,delta-method from per-arm cell SD,0,True
168,CD70,52,1.77,0.146,1.09,1.04,0.981,2.65e+03,0.589,0,-0.344,3.89,0.885,"none -- tau2 absorbs sampling noise, I2 is an upper bound",0,True
1038,TNFRSF1B,52,0.387,0.148,1.1,1.05,0.992,6.36e+03,2.7,0,-1.74,2.51,0.577,delta-method from per-arm cell SD,0,True
1009,TIGIT,52,1.47,0.164,1.34,1.16,0.991,5.51e+03,0.788,0,-0.881,3.82,0.827,delta-method from per-arm cell SD,0,True
773,PI16,52,1.18,0.177,1.58,1.26,0.994,7.98e+03,1.07,0,-1.37,3.73,0.846,delta-method from per-arm cell SD,0,True
265,CTLA4,52,0.649,0.179,1.6,1.27,0.99,4.95e+03,1.95,0,-1.92,3.22,0.731,delta-method from per-arm cell SD,0,True
761,PDCD1,52,1.03,0.195,1.92,1.39,0.991,5.82e+03,1.34,0,-1.78,3.84,0.865,delta-method from per-arm cell SD,0,True
111,CCR4,52,0.885,0.201,2.05,1.43,0.993,7.67e+03,1.62,0,-2.02,3.79,0.769,delta-method from per-arm cell SD,0,True
362,FCRL3,52,1.44,0.208,2.21,1.49,0.981,2.65e+03,1.03,0,-1.58,4.46,0.712,"none -- tau2 absorbs sampling noise, I2 is an upper bound",0,True
1048,TPO,52,2.01,0.223,2.56,1.6,0.998,2.93e+04,0.795,0,-1.23,5.26,0.962,delta-method from per-arm cell SD,0,True
430,GPR15,52,1.54,0.258,3.4,1.84,0.998,2.29e+04,1.2,0,-2.2,5.28,0.827,delta-method from per-arm cell SD,0,True
570,KIR3DL2,52,2.55,0.3,4.63,2.15,0.999,4.58e+04,0.845,0,-1.82,6.91,0.904,delta-method from per-arm cell SD,0,True
```

## §4a Most consistent antigens cohort-wide

**What:** pooled effect above 0.3 with a prediction interval excluding zero.

**How:** same model, ranked by mu.

```csv
index,gene,n_donors,mu,tau,pi_lo,pi_hi,frac_donors_up,on_shortlist
643,MBP,52,0.793,0.321,0.141,1.45,0.942,True
512,IGSF8,52,0.582,0.275,0.021,1.14,0.962,False
```

## §4b COPA, subgroup antigens

**What:** genes extreme in a minority of patients, which a mean test averages away.

**How:** median/MAD-standardised 75th percentile, sign-flip null, BH.

**Full table on disk:** `tables/tgt44_heterogeneity_copa.csv`

```csv
gene,copa_p75,frac_outlier,q_copa,mu,log2fc_paired,q,on_shortlist
GRIK2,35.4,0.173,0.051,0.107,0,0.000234,False
CNTNAP2,29.2,0.135,0.0658,0.0124,0,0.0197,False
EPCAM,19.9,0.173,0.0392,0.737,0.0183,2.08e-05,False
MSR1,17.8,0.192,0.0392,0.295,0.00603,4.62e-05,False
RAMP2,16.3,0.0769,0.0392,0.0646,0,0.00319,False
FCRL2,15.1,0.135,0.127,0.0122,0,0.0174,False
ADAM29,11.6,0.0962,0.0392,0.437,0.0332,5.57e-06,False
CHRNB3,9.65,0.192,0.0392,0.129,0.00235,0.000831,False
CLEC12A,8.85,0.212,0.848,0.00847,0,0.326,False
VNN3,5.6,0.154,0.203,0.00416,0,0.104,False
LILRA6,5.48,0.135,0.0844,0.0314,0,0.00385,False
CYBB,4.86,0.192,0.242,0.0167,0,0.0298,False
S1PR3,4.77,0.192,0.697,0.00142,0,0.443,False
SCN2A,4.57,0.231,0.0392,0.221,0,0.000851,False
BTNL9,4.53,0.154,0.0392,0.122,0.0219,5.57e-06,False
CHRNA7,4.3,0.0962,0.0658,0.468,0.0981,2.23e-06,False
AOC3,3.87,0.115,0.0392,0.0133,0.00196,0.00231,False
LAMA3,3.8,0.135,0.0392,0.15,0.0155,0.000155,False
GALNT9,3.79,0.192,0.0392,0.443,0.0504,2.31e-05,False
SLC6A3,3.6,0.212,0.344,0.0129,0,0.0711,False
SLC8A1,3.59,0.192,0.829,0.0539,0,0.151,False
KIR2DL1,3.52,0.154,0.0392,0.323,0.0143,0.00206,False
GPNMB,3.47,0.135,0.15,0.0118,0.00389,0.00551,False
FFAR3,3.44,0.192,0.0658,0.157,0.017,0.000603,False
PTPRF,3.4,0.154,0.548,0.0068,0,0.114,False
PTGER3,3.33,0.192,0.827,0.0357,0,0.175,False
COL6A6,3.28,0.192,0.15,-0.00101,0,0.0493,False
NRP2,3.11,0.192,0.0946,0.152,0.00173,0.0172,False
DCT,3.11,0.135,1,0.000484,0,0.959,False
KIR3DL1,3.02,0.231,0.0392,0.597,0.0811,0.000726,False
THSD4,3.02,0.212,0.0392,0.0893,0.0176,4.7e-05,False
ABI3BP,2.89,0.192,0.127,0.0185,0,0.0692,False
RAB25,2.8,0,0.343,1.22,0.375,1.1e-05,False
SORT1,2.75,0.173,0.7,0.00998,0,0.191,False
SLC4A1,2.73,0.231,0.203,0.0266,0,0.0315,False
OLFML3,2.7,0.192,0.827,0.0016,0,0.572,False
COL1A1,2.68,0.115,0.0392,0.0674,0.00973,0.00632,False
CLDN7,2.62,0.212,0.051,0.42,9.62e-06,0.202,False
DSG2,2.54,0.231,0.681,0.102,0,0.0403,False
NTRK2,2.53,0.154,0.051,0.286,0.0266,0.00115,False
```

## §4b Bimodality of the per-donor effect

**What:** whether a gene's donor effects fall into two groups rather than one spread.

**How:** one- versus two-component Gaussian mixture BIC; delta_bic > 0 favours two components. A weak instrument at this donor count -- read it beside I2 and COPA.

**Full table on disk:** `tables/tgt44_heterogeneity_bimodality.csv`

```csv
gene,n_donors,delta_bic,two_component,mode_lo,mode_hi,w_hi,bimodality_coef,copa_p75,frac_outlier
SLCO6A1,52,399,True,1.76e-19,0.164,0.0965,0.887,0,0
LRRC15,52,385,True,-0.22,0,0.981,0.96,0,0
SLITRK6,52,346,True,1.21e-54,0.102,0.0774,0.893,0,0
TMEM132B,52,339,True,2.03e-42,0.071,0.0969,0.799,0,0
EFNA2,52,339,True,4.66e-05,0.101,0.116,0.907,0,0
ANGPTL1,52,326,True,-0.184,3.27e-85,0.73,0.841,0,0
PDGFRB,52,314,True,-0.181,8.83e-26,0.691,0.764,0,0
PVRL2,52,309,True,-0.084,6.44e-05,0.902,0.863,0,0
MMP16,52,308,True,0.000663,0.297,0.178,0.866,0,0
GJB2,52,306,True,0.00084,0.301,0.234,0.891,0,0
ADRA2A,52,299,True,-0.0492,7.73e-05,0.876,0.828,0,0
CD177,52,295,True,0.00278,0.737,0.201,0.902,0,0
SIGLEC12,52,288,True,-7.28e-05,0.0786,0.234,0.823,0,0
ST14,52,287,True,3.89e-05,0.575,0.462,0.81,0,0
GLDN,52,287,True,3.1e-10,0.0711,0.213,0.74,0,0
CACNA2D1,52,285,True,-0.0301,8.02e-10,0.864,0.833,0,0
CLEC17A,52,271,True,0.000106,0.129,0.31,0.812,0,0
OR2AG1,52,271,True,-0.0959,8.27e-05,0.981,0.958,0,0
OTOP2,52,263,True,0,0.068,0.0192,0.96,0,0
COL2A1,52,262,True,-0.0395,4.92e-05,0.905,0.823,0,0
OCA2,52,260,True,-0.0309,2.46e-05,0.786,0.778,0,0
CRB2,52,256,True,-0.0539,0.000114,0.687,0.84,0,0
ASGR2,52,255,True,0.000297,0.0359,0.187,0.783,0,0
PARM1,52,254,True,0.00104,0.567,0.334,0.829,0,0
TLR4,52,253,True,1.61e-06,0.0202,0.252,0.42,0,0
```

## §4c Donor clusters against the clinical fields

**What:** whether the patients fall into groups on these antigens.

**How:** hierarchical clustering of the donor x gene matrix, silhouette cut, min 3 donors.

**Full table on disk:** `tables/tgt44_donor_clusters.csv`

```csv
donor,dx2,study,sex,age,burden,tx_ecp,tx_hdaci,is_cultured,n_prior_lines,cluster
B1__P4,SS,borcherding2019,M,61,0.982,True,True,False,,1
B2__CTCL4,SS,borcherding2023,F,84,0.577,True,False,False,,1
B2__CTCL5,SS,borcherding2023,M,73,0.739,True,False,False,,1
B2__CTCL6,SS,borcherding2023,F,69,0.495,False,True,False,,1
B4__SZ30_HTO6,unresolved,geskin2026,M,70,0.646,False,True,True,,1
B4__SZ32_HTO1,SS,geskin2026,M,32,0.142,False,False,True,,1
B4__SZ38_HTO1,SS,geskin2026,M,,0.994,False,False,True,,1
B4__SZ39_HTO1,SS,geskin2026,F,,0.557,False,False,True,,1
B4__SZ5_HTO1,SS,geskin2026,F,,0.985,False,False,True,,1
B5__PT02,SS,buus2025,M,,0.993,True,False,False,,1
B5__PT05,SS,buus2025,F,,0.979,True,False,False,,1
B5__PT06,SS,buus2025,F,79,0.974,True,False,False,,1
B5__PT07,SS,buus2025,F,60,0.973,True,False,False,,1
B5__PT08,SS,buus2025,M,75,0.969,False,True,False,,1
B5__PT09,SS,buus2025,F,71,0.946,True,False,False,,1
B5__PT10,SS,buus2025,M,79,0.953,True,False,False,,1
B5__PT13,SS,buus2025,M,80,0.935,True,False,False,,1
B5__PT14,SS,buus2025,M,57,0.933,False,False,False,,1
B5__PT15,SS,buus2025,M,,0.91,True,False,False,,1
B5__PT16,SS,buus2025,M,67,0.961,True,False,False,,1
B5__PT18,MF,buus2025,F,,0.791,True,False,False,,1
B5__PT21,MF,buus2025,M,70,0.296,False,True,False,,1
B5__PT22,SS,buus2025,M,70,0.22,False,False,False,,1
B5__PT23,SS,buus2025,F,85,0.333,True,False,False,,1
B6__P1,SS,ren2023,M,81,0.979,True,False,False,,1
B6__P10,MF,ren2023,M,70,0.707,True,False,False,,1
B6__P11,MF,ren2023,M,76,0.784,True,False,False,,1
B6__P2,MF,ren2023,M,76,0.745,True,False,False,,1
B6__P3,SS,ren2023,F,72,0.901,True,False,False,,1
B6__P4,MF,ren2023,M,79,0.714,True,False,False,,1
B6__P5,SS,ren2023,M,56,0.723,True,False,False,,1
B6__P6,SS,ren2023,M,85,0.954,True,False,False,,1
B6__P7,SS,ren2023,F,78,0.817,True,True,False,,1
B6__P9,SS,ren2023,F,68,0.923,True,False,False,,1
B7__SS1,SS,harro2023,F,76,0.908,False,False,False,,1
B7__SS2,SS,harro2023,M,80,0.979,False,False,False,,1
B7__SS4,SS,harro2023,M,70,0.927,False,False,False,,1
B8__1084,unresolved,dorando2026,F,74,0.982,False,False,False,0,1
B8__1126,unresolved,dorando2026,F,61,0.941,False,False,False,2,1
B8__1135,unresolved,dorando2026,M,49,0.994,False,False,False,2,1
B8__1181,unresolved,dorando2026,M,59,0.1,False,False,False,4,1
B8__999,unresolved,dorando2026,F,,0.898,False,False,False,,1
D13__SZ16,SS,gaydosik2022,M,56,0.05,False,False,False,,1
D13__SZ22,unresolved,gaydosik2022,M,62,0.319,False,False,False,,1
D13__SZ29,SS,gaydosik2022,M,68,0.955,False,False,False,,1
D5__MFIVB,MF,rindler2021_fi,F,65,0.419,False,False,False,,1
H__MFIV1,MF,herrera2021,F,39,0.499,False,False,False,,1
H__SS1,SS,herrera2021,F,49,0.601,False,False,False,,1
H__SS3,SS,herrera2021,F,71,0.771,False,False,False,,1
H__SS4,SS,herrera2021,M,56,0.33,False,False,False,,1
H__SS5,SS,herrera2021,F,71,0.578,True,False,False,,1
H__SS6,SS,herrera2021,M,78,0.599,False,False,False,,1
```

## §4c Per-cluster mean effect for the leading antigens

**What:** what separates the donor clusters, where there are any.

**How:** mean per-donor effect inside each cluster; `spread` is max minus min across clusters.

**Full table on disk:** `tables/tgt44_donor_cluster_profiles.csv`

```csv
gene,1,spread
KIR3DL2,2.55,0
CD70,1.77,0
TPO,2.01,0
TIGIT,1.47,0
STAM,1.44,0
FCRL3,1.44,0
GPR15,1.53,0
PDCD1,1.03,0
NRROS,1.11,0
PI16,1.17,0
ITGB1,1.08,0
CCR4,0.88,0
CD84,1.13,0
EPHA4,1.07,0
TSPAN2,1.14,0
ABCC1,0.871,0
MBP,0.785,0
GPR146,0.855,0
FUT8,0.695,0
SLAMF6,0.831,0
CD82,0.758,0
TFRC,0.814,0
IL6R,0.762,0
ANXA2,0.651,0
TNFRSF1B,0.394,0
```

## §4e Eligibility

**What:** the fraction of patients in whom at least X% of malignant CD4 cells are positive.

**How:** transcript detection per donor, thresholded at 5/10/25/50/75%.

```csv
index,gene,n_donors,median_pct_expr,frac_donors_ge5,frac_donors_ge10,frac_donors_ge25,frac_donors_ge50,frac_donors_ge75,mu,I2,on_shortlist
4,IL6R,52,0.243,1,0.981,0.481,0.173,0.0192,0.754,0.972,True
38,MBP,52,0.719,1,1,1,0.885,0.423,0.793,0.965,True
47,FUT8,52,0.227,1,0.885,0.404,0.0385,0,0.712,0.955,True
81,CD84,52,0.243,1,0.942,0.481,0.0577,0.0192,1.13,0.976,True
84,SLAMF6,52,0.25,1,0.885,0.5,0.0769,0,0.827,0.959,True
88,CD28,52,0.412,1,1,0.846,0.212,0.0192,0.578,0.96,True
98,ANXA2,52,0.524,1,0.981,0.885,0.538,0.269,0.648,0.99,True
100,TNFRSF25,52,0.395,1,1,0.712,0.327,0.0385,0.422,0.984,True
102,ABCC1,52,0.221,1,0.942,0.442,0.0577,0,0.893,0.928,True
116,EPHA4,52,0.189,0.981,0.846,0.308,0.0192,0,1.08,0.977,True
124,ITGB1,52,0.828,0.981,0.981,0.981,0.923,0.673,1.08,0.995,True
129,CD82,52,0.44,0.962,0.923,0.808,0.385,0.0385,0.763,0.982,True
133,TIGIT,52,0.424,0.962,0.923,0.75,0.327,0.154,1.47,0.991,True
135,SIRPG,52,0.245,0.962,0.827,0.442,0.154,0,0.584,0.974,True
139,STAM,52,0.445,0.962,0.885,0.769,0.404,0.0192,1.45,0.979,True
144,TFRC,52,0.134,0.942,0.673,0.115,0,0,0.814,0.949,True
145,TNFRSF1B,52,0.291,0.942,0.885,0.654,0.288,0.0962,0.387,0.992,True
148,CTLA4,52,0.159,0.923,0.75,0.288,0.0769,0.0192,0.649,0.99,True
153,GRN,52,0.159,0.904,0.673,0.192,0,0,0.637,0.931,True
154,CD70,52,0.241,0.904,0.808,0.481,0.0577,0,1.77,0.981,True
155,TSPAN2,52,0.132,0.904,0.615,0.288,0.0962,0,1.18,0.978,True
164,NRROS,52,0.149,0.865,0.615,0.308,0,0,1.11,0.976,True
176,CCR4,52,0.231,0.827,0.731,0.462,0.115,0,0.885,0.993,True
178,PI16,52,0.254,0.808,0.712,0.5,0.25,0.0192,1.18,0.994,True
182,KIR3DL2,52,0.367,0.808,0.769,0.615,0.423,0.115,2.55,0.999,True
186,PDCD1,52,0.113,0.788,0.519,0.192,0.0192,0,1.03,0.991,True
190,GPR146,52,0.0985,0.769,0.5,0.192,0.0192,0,0.862,0.966,True
214,TPO,52,0.121,0.673,0.558,0.365,0.173,0.0192,2.01,0.998,True
215,FCRL3,52,0.198,0.673,0.596,0.385,0.115,0,1.44,0.981,True
220,GPR15,52,0.182,0.635,0.558,0.385,0.115,0.0192,1.54,0.998,True
```

## §4f Large and consistent vs large and selective

**What:** the two quadrants a development decision turns on.

**How:** mu > 0.5 split on tau against its cohort median and on whether the prediction interval excludes zero.

```csv
index,gene,quadrant,mu,tau,pi_lo,pi_hi,frac_donors_up,on_shortlist
0,KIR3DL2,large_but_variable,2.55,2.15,-1.82,6.91,0.904,True
2,TPO,large_but_variable,2.01,1.6,-1.23,5.26,0.962,True
3,GPR15,large_but_variable,1.54,1.84,-2.2,5.28,0.827,True
16,ITGB1,large_but_variable,1.08,0.859,-0.663,2.82,0.904,True
23,PI16,large_but_variable,1.18,1.26,-1.37,3.73,0.846,True
26,CCR4,large_but_variable,0.885,1.43,-2.02,3.79,0.769,True
36,PDCD1,large_but_variable,1.03,1.39,-1.78,3.84,0.865,True
42,TIGIT,large_but_variable,1.47,1.16,-0.881,3.82,0.827,True
46,NT5E,large_but_variable,0.653,0.784,-0.939,2.24,0.615,False
49,CTLA4,large_but_variable,0.649,1.27,-1.92,3.22,0.731,True
52,ANXA2,large_but_variable,0.648,0.71,-0.793,2.09,0.769,True
58,CPM,large_but_variable,1.22,0.999,-0.804,3.25,0.904,False
60,ICAM2,large_but_variable,0.548,0.649,-0.768,1.86,0.75,False
63,GPR55,large_but_variable,1.01,0.828,-0.669,2.69,0.904,False
78,ACVR2A,large_but_variable,0.951,0.942,-0.961,2.86,0.923,False
84,CD82,large_but_variable,0.763,0.664,-0.585,2.11,0.846,True
87,F5,large_but_variable,0.565,1.09,-1.64,2.77,0.692,False
128,KIR3DL1,large_but_variable,0.597,1.07,-1.56,2.76,0.615,False
137,LAG3,large_but_variable,0.69,1.17,-1.69,3.07,0.75,False
164,NXPH4,large_but_variable,0.736,0.888,-1.06,2.54,0.865,False
180,LRRC32,large_but_variable,0.543,0.854,-1.19,2.27,0.558,False
222,IGHM,large_but_variable,0.633,0.663,-0.712,1.98,0.788,False
344,PTH2R,large_but_variable,1.16,1.1,-1.07,3.4,0.865,False
360,RAB25,large_but_variable,1.22,1.46,-1.74,4.18,0.769,False
422,MRC1,large_but_variable,0.535,0.735,-0.955,2.02,0.731,False
456,NRN1,large_but_variable,0.732,0.819,-0.93,2.39,0.769,False
488,CDH1,large_but_variable,0.827,0.94,-1.08,2.73,0.731,False
513,CD70,large_but_variable,1.77,1.04,-0.344,3.89,0.885,True
539,CSF2RB,large_but_variable,0.609,0.761,-0.935,2.15,0.75,False
716,FZD3,large_but_variable,0.57,0.945,-1.35,2.49,0.673,False
793,FCRL3,large_but_variable,1.44,1.49,-1.58,4.46,0.712,True
811,ENTPD1,large_but_variable,0.562,0.962,-1.39,2.51,0.692,False
812,EPCAM,large_but_variable,0.737,1.2,-1.7,3.17,0.596,False
851,FGL2,large_but_variable,0.608,0.753,-0.919,2.14,0.769,False
866,ST6GALNAC1,large_but_variable,0.587,1.09,-1.62,2.8,0.654,False
884,SLAMF1,large_but_variable,0.6,0.828,-1.08,2.28,0.788,False
887,ADGRE1,large_but_variable,1.05,0.839,-0.656,2.75,0.865,False
894,FLNA,large_but_variable,0.52,0.41,-0.312,1.35,0.846,False
895,STAM,large_but_variable,1.45,0.836,-0.249,3.14,0.865,True
897,PTGER4,large_but_variable,0.538,0.642,-0.764,1.84,0.808,False
898,FGFR1,large_but_variable,1.09,0.841,-0.622,2.79,0.885,False
900,TSPAN2,large_but_variable,1.18,0.859,-0.567,2.92,0.923,True
902,EPHA4,large_but_variable,1.08,0.905,-0.756,2.92,0.942,True
903,NRROS,large_but_variable,1.11,0.823,-0.561,2.78,0.904,True
905,CD84,large_but_variable,1.13,0.751,-0.394,2.65,0.981,True
912,SIRPG,large_but_variable,0.584,0.534,-0.5,1.67,0.865,True
913,GPR155,large_but_variable,0.759,0.788,-0.84,2.36,0.827,False
919,IL6R,large_but_variable,0.754,0.605,-0.474,1.98,0.923,True
940,GPR146,large_but_variable,0.862,0.679,-0.516,2.24,0.904,True
947,MBP,large_but_variable,0.793,0.321,0.141,1.45,0.942,True
[9 further rows omitted from this block]
```

======================== COPY TO HERE ==========================