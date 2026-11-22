> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 47 — Skin stratification and heterogeneity

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**. This notebook stratifies the malignant-vs-reactive CD4 antigen
axis that notebook 42 established, and measures how much it varies between patients. It derives
no new screen: it reads 42's cleaned cohort and tables.

Cohort: 75 donors (75 people, 12 studies)
carrying BOTH a malignant and a reactive CD4 arm of >= 20 cells. Non-CD4 clones, diagnoses
outside the MF/SS spectrum and repeat biopsies of one patient were already removed in 42 §0b.
Diagnosis from the curated `clin_dx`.

Per-donor effect: for each gene and donor, log2((mean malignant + 0.05)/(mean reactive + 0.05))
on log1p CP10K. Everything below is a function of that donor x gene matrix.

Stratified tests: the paired Wilcoxon is rerun inside each stratum, and `p_interaction` compares
the per-donor effects between strata (BH-adjusted as `q_interaction`). Per-gene OLS regresses the
donor effect on lesion class, diagnosis, stage, sex, age and burden at once.

Heterogeneity: DerSimonian-Laird random effects over donors (`mu` pooled effect, `tau`
between-patient SD, `I2`, `pi_lo`/`pi_hi` prediction interval, `q_Q` BH-adjusted Q test); COPA and
the outlier sum with a 1000-draw sign-flip null; two-component Gaussian mixture BIC;
hierarchical donor clustering with a silhouette cut and a minimum of 3 donors per cluster.

What is special about the skin cohort relative to the blood one (notebook 44): the lesion that was
biopsied is recorded (patch / plaque / tumour / erythroderma), and stage is testable here
(early vs advanced) where blood had 2 early-stage donors. The lesion is confounded with the
malignant burden -- a tumour holds a larger clone than a patch -- and that confounding is measured
rather than assumed.

## §1 Power table

**What:** coverage of every candidate stratification over the paired donors.

**How:** count of non-missing values per field, split by diagnosis; `levels` gives the value counts or the numeric range.

**Full table on disk:** `tables/tgt47_power_table.csv`

```csv
field,n,MF,SS,levels
dx2 (MF vs SS),75,67,6,"MF:67, SS:6, unresolved:2"
lesion,75,67,6,"tumor:19, mixed:17, erythroderma:11, plaque:10, patch:9, skin_unspecified:8"
lesion class,75,67,6,"lesion:67, unspecified:8"
sex,75,67,6,"M:44, F:31"
age,75,67,6,"median 64, range 30-87"
stage,73,67,4,"IIB:30, IB:11, IVA1:8, IA:7, IVB:4, IVA2:4"
stage class,73,67,4,"advanced:55, early:18"
B stage,57,51,4,"B0:40, B2:9, B1:5, B2b:3"
CD30,17,17,0,"pos:9, neg:8"
large-cell transformation,32,32,0,"yes:23, no:9"
folliculotropic,5,5,0,yes:5
timepoint,75,67,6,"single:60, pre:5, baseline:5, serial:4, on_treatment:1"
treatment at sampling known,51,43,6,"median 1, range 0-1"
treatment-naive,58,50,6,"no:46, yes:6, washout_6mo:6"
prior lines,0,0,0,
response,7,6,1,"PD:3, cleared:2, NR:1, SD:1"
outcome,19,19,0,"alive:13, DOD:5, died_unspecified:1"
malignant burden,75,67,6,"median 0.658, range 0.00837-0.985"
```

## §1 Treatment at sampling

**What:** donors on each treatment class at the time of sampling, and in which studies.

**How:** membership of the curated clin_current_tx_classes string. A donor can be on several.

```csv
index,donors,MF,SS,studies
ecp,7,5,2,"brunner2024, buus2025, chennareddy2025, rindler2021_mc"
retinoid,4,3,1,"chennareddy2025, gaydosik2019, il4ra2026"
interferon,2,2,0,"brunner2024, chennareddy2025"
methotrexate,2,2,0,il4ra2026
topical_steroid,2,1,1,"gaydosik2019, il4ra2026"
hdaci,1,0,1,gaydosik2019
brentuximab,1,1,0,brentuximab2026
mogamulizumab,1,1,0,gaydosik2022
alemtuzumab,1,0,1,buus2025
other_immunomod,1,1,0,jonak2021
phototherapy,1,1,0,il4ra2026
```

## §2a Lesion class, burden confounding

**What:** the malignant share of the CD4 compartment inside each lesion class -- the confounder of the lesion contrast.

**How:** burden = malignant cells / (malignant + reactive) per donor.

```csv
lesion,count,mean,std,min,25%,50%,75%,max
patch/plaque,19,0.544,0.234,0.153,0.329,0.566,0.752,0.883
tumour/eryth,30,0.589,0.291,0.00837,0.359,0.624,0.827,0.985
```

## §2a Lesion class, 42's shortlist

**What:** each leading antigen's malignant-vs-reactive effect inside patch/plaque and inside tumour/erythroderma, and whether the two differ.

**How:** paired Wilcoxon within each stratum; delta_lfc and q_interaction compare the per-donor effects between strata.

**Full table on disk:** `tables/tgt47_strat_lesion.csv`

```csv
index,gene,log2fc_paired_patch_plaque,log2fc_paired_tumour_eryth,n_donors_paired_patch_plaque,n_donors_paired_tumour_eryth,delta_lfc,q_interaction,log2fc_all,on_shortlist
114,AP2S1,0.547,0.52,19,30,-0.0278,0.999,0.579,True
228,CALU,0.587,0.493,19,30,-0.0936,0.897,0.587,True
256,CD151,0.988,0.103,19,30,-0.885,0.74,0.794,True
313,CD70,1.09,1.1,19,30,0.0132,1,1.1,True
319,CD82,0.609,0.593,19,30,-0.0161,1,0.632,True
325,CD9,1.08,0.946,19,30,-0.139,1,1.08,True
471,COL6A3,1.17,0.842,19,30,-0.328,0.898,0.811,True
478,CPM,1.39,1.39,19,30,0.00716,1,1.02,True
718,FUT8,0.659,0.676,19,30,0.0166,1,0.635,True
807,GPI,0.595,0.49,19,30,-0.105,1,0.507,True
888,GRN,0.953,0.623,19,30,-0.33,0.88,0.778,True
1003,IL6R,0.687,0.844,19,30,0.157,0.933,0.614,True
1117,LGALS3,1.06,0.0216,19,30,-1.04,1,0.663,True
1637,PDIA4,0.919,0.48,19,30,-0.439,0.65,0.637,True
1745,QSOX1,0.793,0.329,19,30,-0.464,0.777,0.584,True
1928,SLC1A4,0.307,0.724,19,30,0.416,1,0.69,True
1976,SLC39A6,0.84,0.512,19,30,-0.328,0.678,0.707,True
1993,SLC4A2,0.763,0.537,19,30,-0.226,0.876,0.698,True
2004,SLC5A3,0.846,0.514,19,30,-0.332,0.785,0.617,True
2104,STX6,0.633,0.583,19,30,-0.0502,0.929,0.578,True
2112,SURF4,0.8,0.166,19,30,-0.634,0.59,0.552,True
2117,SYPL1,0.378,0.604,19,30,0.226,0.949,0.599,True
2138,TFRC,1.07,0.693,19,30,-0.38,0.704,0.731,True
2196,TMEM30A,0.694,0.237,19,30,-0.457,0.497,0.506,True
2227,TNFRSF25,0.947,0.722,19,30,-0.224,0.999,0.698,True
2228,TNFRSF4,0.674,0.45,19,30,-0.223,0.989,0.542,True
2268,TSPAN2,0.53,0.623,19,30,0.0928,1,0.53,True
2315,VDAC1,0.536,0.51,19,30,-0.0253,1,0.584,True
```

## §2b Disease, MF vs SS

**What:** each leading antigen's malignant-vs-reactive effect inside each level of disease (MF vs SS), and whether the levels differ.

**How:** paired Wilcoxon within each stratum; delta_lfc and q_interaction compare the per-donor effects between strata.

**Full table on disk:** `tables/tgt47_strat_disease.csv`

```csv
index,gene,log2fc_paired_MF,log2fc_paired_SS,n_donors_paired_MF,n_donors_paired_SS,delta_lfc,q_interaction,on_shortlist
114,AP2S1,0.621,0.376,67,6,0.245,0.312,True
228,CALU,0.601,0.381,67,6,0.219,0.847,True
256,CD151,0.988,-0.147,67,6,1.13,0.0409,True
313,CD70,1.12,0.997,67,6,0.119,0.898,True
319,CD82,0.649,0.36,67,6,0.29,0.549,True
325,CD9,1.31,-0.194,67,6,1.5,0.108,True
471,COL6A3,0.957,0.842,67,6,0.115,0.898,True
478,CPM,0.951,1.86,67,6,-0.908,0.497,True
718,FUT8,0.635,0.631,67,6,0.00386,0.822,True
807,GPI,0.545,-0.105,67,6,0.649,0.113,True
888,GRN,0.821,0.474,67,6,0.347,0.759,True
1003,IL6R,0.613,0.95,67,6,-0.337,0.742,True
1117,LGALS3,1.06,-0.572,67,6,1.64,0.157,True
1637,PDIA4,0.747,0.0445,67,6,0.703,0.0409,True
1745,QSOX1,0.661,-0.0214,67,6,0.683,0.0773,True
1928,SLC1A4,0.73,-0.258,67,6,0.988,0.295,True
1976,SLC39A6,0.744,0.358,67,6,0.386,0.386,True
1993,SLC4A2,0.763,0.578,67,6,0.185,0.601,True
2004,SLC5A3,0.628,0.776,67,6,-0.149,0.972,True
2104,STX6,0.583,0.546,67,6,0.0365,1,True
2112,SURF4,0.592,-0.181,67,6,0.773,0.0611,True
2117,SYPL1,0.599,0.181,67,6,0.418,0.764,True
2138,TFRC,0.813,0.662,67,6,0.151,0.87,True
2196,TMEM30A,0.51,0.00823,67,6,0.502,0.429,True
2227,TNFRSF25,0.675,1.03,67,6,-0.354,0.378,True
2228,TNFRSF4,0.437,1.2,67,6,-0.766,0.424,True
2268,TSPAN2,0.552,0.352,67,6,0.2,0.934,True
2315,VDAC1,0.71,0.284,67,6,0.427,0.361,True
```

## §2b Stage class, early vs advanced

**What:** each leading antigen's malignant-vs-reactive effect inside each level of stage class, and whether the levels differ.

**How:** paired Wilcoxon within each stratum; delta_lfc and q_interaction compare the per-donor effects between strata.

**Full table on disk:** `tables/tgt47_strat_stage.csv`

```csv
index,gene,log2fc_paired_advanced,log2fc_paired_early,n_donors_paired_advanced,n_donors_paired_early,delta_lfc,q_interaction,on_shortlist
114,AP2S1,0.553,0.695,55,18,-0.142,0.564,True
228,CALU,0.543,0.731,55,18,-0.188,1,True
256,CD151,0.794,1.06,55,18,-0.266,0.707,True
313,CD70,0.958,1.49,55,18,-0.529,0.274,True
319,CD82,0.632,0.706,55,18,-0.0734,0.904,True
325,CD9,1.03,1.34,55,18,-0.313,0.757,True
471,COL6A3,0.957,0.453,55,18,0.504,0.622,True
478,CPM,1.2,0.684,55,18,0.511,0.696,True
718,FUT8,0.635,0.584,55,18,0.0517,0.853,True
807,GPI,0.518,0.551,55,18,-0.0331,1,True
888,GRN,0.821,0.715,55,18,0.106,1,True
1003,IL6R,0.649,0.108,55,18,0.541,0.257,True
1117,LGALS3,0.119,1.49,55,18,-1.37,0.104,True
1637,PDIA4,0.659,0.669,55,18,-0.0101,1,True
1745,QSOX1,0.572,0.656,55,18,-0.0842,0.654,True
1928,SLC1A4,0.42,1.63,55,18,-1.21,0.088,True
1976,SLC39A6,0.702,0.843,55,18,-0.141,0.611,True
1993,SLC4A2,0.625,0.783,55,18,-0.158,0.981,True
2004,SLC5A3,0.548,0.551,55,18,-0.00272,1,True
2104,STX6,0.583,0.555,55,18,0.0281,1,True
2112,SURF4,0.437,0.787,55,18,-0.35,0.306,True
2117,SYPL1,0.66,0.344,55,18,0.316,0.586,True
2138,TFRC,0.735,0.612,55,18,0.123,0.815,True
2196,TMEM30A,0.398,0.739,55,18,-0.341,0.116,True
2227,TNFRSF25,0.749,0.0497,55,18,0.699,0.564,True
2228,TNFRSF4,0.542,0.524,55,18,0.0179,0.897,True
2268,TSPAN2,0.565,0.434,55,18,0.13,0.461,True
2315,VDAC1,0.598,0.586,55,18,0.012,1,True
```

## §2c Donor-level regression

**What:** which donor properties move each gene's effect, with all terms in one model.

**How:** OLS per gene over donors, terms is_tumour_lesion, is_MF, is_advanced, is_F, age_z, burden_z; BH per term. The term correlation matrix is given because lesion and burden share variance.

**Full table on disk:** `tables/tgt47_strat_regression.csv`

```csv
key,genes_at_q_lt_0.05
is_tumour_lesion,0
is_MF,0
is_advanced,0
is_F,0
age_z,0
burden_z,0
```

## §2c Term correlations

**What:** how far the model's terms overlap.

**How:** Pearson between the donor-level covariates.

```csv
index,is_tumour_lesion,is_MF,is_advanced,is_F,age_z,burden_z
is_tumour_lesion,1,-0.335,0.369,0.0884,0.212,-0.0221
is_MF,-0.335,1,-0.013,-0.0608,-0.0801,-0.0086
is_advanced,0.369,-0.013,1,-0.106,0.0288,0.165
is_F,0.0884,-0.0608,-0.106,1,-0.167,0.115
age_z,0.212,-0.0801,0.0288,-0.167,1,0.0229
burden_z,-0.0221,-0.0086,0.165,0.115,0.0229,1
```

## §3 Tier 2 descriptive splits

**What:** effect sizes on each side of the splits too small to test -- CD30, large-cell transformation, folliculotropism, B stage, individual treatments.

**How:** median per-donor effect per side with a 400-draw bootstrap CI. No p-value, no correction.

**Full table on disk:** `tables/tgt47_tier2_descriptive.csv`

```csv
split,gene,n_yes,n_no,median_yes,median_no,delta,ci_lo,ci_hi
CD30 positive,CD70,9,8,1.12,0.377,0.739,-0.0416,2.25
CD30 positive,CD9,9,8,1.08,1.76,-0.675,-2.44,2.38
CD30 positive,CPM,9,8,1.72,1.25,0.474,-2.34,2.57
CD30 positive,COL6A3,9,8,1.49,0.468,1.02,-0.0731,2.41
CD30 positive,CD151,9,8,1.33,1.01,0.318,0.0425,1.28
CD30 positive,GRN,9,8,1.02,0.872,0.15,-0.802,0.646
CD30 positive,TFRC,9,8,1.01,1.07,-0.0586,-0.884,0.884
CD30 positive,SLC39A6,9,8,0.725,0.919,-0.194,-0.538,0.45
CD30 positive,SLC4A2,9,8,1,0.724,0.276,-0.65,0.971
CD30 positive,TNFRSF25,9,8,0.947,0.917,0.0294,-0.686,2.1
CD30 positive,SLC1A4,9,8,0.307,0.604,-0.297,-2.27,0.63
CD30 positive,LGALS3,9,8,-0.256,1.08,-1.34,-2.49,2.2
CD30 positive,PDIA4,9,8,0.938,1.23,-0.289,-0.808,0.359
CD30 positive,FUT8,9,8,1.01,0.728,0.281,-0.574,1.39
CD30 positive,CD82,9,8,1.06,0.678,0.38,-1.12,0.851
large-cell transformation,CD70,23,9,1.09,0.412,0.68,-1.47,1.08
large-cell transformation,CD9,23,9,1.69,1.33,0.359,-1.12,2.38
large-cell transformation,CPM,23,9,1.46,0.951,0.512,-1.19,1.8
large-cell transformation,COL6A3,23,9,1.49,1.16,0.33,-0.0165,1.18
large-cell transformation,CD151,23,9,1.15,0.96,0.187,-0.0368,1.32
large-cell transformation,GRN,23,9,0.923,0.379,0.544,-0.302,0.864
large-cell transformation,TFRC,23,9,0.962,0.815,0.147,-0.73,0.696
large-cell transformation,SLC39A6,23,9,0.725,0.651,0.0735,-0.429,0.602
large-cell transformation,SLC4A2,23,9,1.22,0.415,0.81,0.0439,1.11
large-cell transformation,TNFRSF25,23,9,0.698,1.2,-0.505,-1.01,1.26
large-cell transformation,SLC1A4,23,9,0.218,1.61,-1.39,-2.04,0.179
large-cell transformation,LGALS3,23,9,1.01,1.2,-0.184,-1.91,1.66
large-cell transformation,PDIA4,23,9,0.98,0.825,0.155,-0.661,0.54
large-cell transformation,FUT8,23,9,0.642,0.659,-0.017,-0.758,0.77
large-cell transformation,CD82,23,9,0.675,0.632,0.0426,-0.685,0.858
B2 (vs B0/B1),CD70,9,48,0.485,1.07,-0.589,-0.827,0.819
B2 (vs B0/B1),CD9,9,48,-0.163,1.29,-1.46,-2.94,2.15
B2 (vs B0/B1),CPM,9,48,1.14,1.41,-0.267,-1.46,0.827
B2 (vs B0/B1),COL6A3,9,48,0.771,1.17,-0.395,-0.797,0.55
B2 (vs B0/B1),CD151,9,48,-0.113,1.04,-1.15,-1.29,0.416
B2 (vs B0/B1),GRN,9,48,0.515,0.799,-0.285,-0.673,0.657
B2 (vs B0/B1),TFRC,9,48,0.631,0.898,-0.267,-0.668,0.122
B2 (vs B0/B1),SLC39A6,9,48,0.601,0.716,-0.115,-0.896,0.457
B2 (vs B0/B1),SLC4A2,9,48,0.566,0.76,-0.194,-0.66,0.668
B2 (vs B0/B1),TNFRSF25,9,48,0.812,0.751,0.0606,-1.52,0.636
B2 (vs B0/B1),SLC1A4,9,48,-0.073,0.71,-0.783,-1.66,-0.124
B2 (vs B0/B1),LGALS3,9,48,-0.974,1.06,-2.04,-3.61,-0.605
B2 (vs B0/B1),PDIA4,9,48,0.408,0.757,-0.349,-0.778,0.233
B2 (vs B0/B1),FUT8,9,48,0.568,0.676,-0.108,-0.512,0.207
B2 (vs B0/B1),CD82,9,48,0.359,0.662,-0.303,-1.07,0.0825
any systemic treatment at sampling,CD70,15,36,1.16,1.09,0.0647,-0.693,0.931
any systemic treatment at sampling,CD9,15,36,0.471,1.08,-0.607,-1.92,1.32
any systemic treatment at sampling,CPM,15,36,1.59,1.11,0.48,-0.336,1.68
any systemic treatment at sampling,COL6A3,15,36,0.994,0.776,0.218,-0.528,0.813
any systemic treatment at sampling,CD151,15,36,0.524,0.951,-0.427,-1.19,0.914
any systemic treatment at sampling,GRN,15,36,0.499,0.828,-0.328,-0.518,0.352
any systemic treatment at sampling,TFRC,15,36,0.691,0.895,-0.203,-0.707,0.104
any systemic treatment at sampling,SLC39A6,15,36,0.576,0.705,-0.129,-0.47,0.174
any systemic treatment at sampling,SLC4A2,15,36,0.439,0.568,-0.129,-0.621,0.398
any systemic treatment at sampling,TNFRSF25,15,36,0.754,0.836,-0.0816,-0.478,0.62
any systemic treatment at sampling,SLC1A4,15,36,0.135,0.836,-0.7,-1.55,-0.184
any systemic treatment at sampling,LGALS3,15,36,-0.459,1.04,-1.5,-1.89,0.159
any systemic treatment at sampling,PDIA4,15,36,0.418,0.709,-0.291,-0.771,0.416
any systemic treatment at sampling,FUT8,15,36,0.866,0.659,0.207,-0.275,0.623
any systemic treatment at sampling,CD82,15,36,0.619,0.679,-0.0602,-0.699,0.233
[75 further rows omitted from this block]
```

## §4a Random-effects meta-analysis, 42's shortlist

**What:** pooled effect and between-patient spread for each leading antigen.

**How:** DerSimonian-Laird; `pi_lo > 0` is what 'works in every patient' means. I2 saturates near 1 because each donor contributes many cells, so rank on tau.

**Full table on disk:** `tables/tgt47_heterogeneity_meta.csv`

```csv
index,gene,n_donors,mu,se,tau,tau_over_mu,I2,pi_lo,pi_hi,frac_donors_up,q_Q,on_shortlist
2104,STX6,75,0.499,0.0442,0.316,0.635,0.84,-0.138,1.14,0.893,1.17e-57,True
2315,VDAC1,75,0.648,0.0429,0.341,0.527,0.956,-0.0378,1.33,0.947,2.14e-302,True
2196,TMEM30A,75,0.433,0.0496,0.376,0.868,0.91,-0.322,1.19,0.787,1.37e-126,True
114,AP2S1,75,0.57,0.0529,0.423,0.742,0.958,-0.279,1.42,0.907,0,True
228,CALU,75,0.614,0.0549,0.423,0.689,0.92,-0.236,1.46,0.88,1.55e-146,True
1993,SLC4A2,75,0.796,0.058,0.443,0.557,0.906,-0.0948,1.69,0.933,1.53e-120,True
2117,SYPL1,75,0.617,0.0557,0.446,0.723,0.961,-0.279,1.51,0.907,0,True
2112,SURF4,75,0.449,0.0557,0.458,1.02,0.975,-0.469,1.37,0.72,0,True
807,GPI,75,0.44,0.0562,0.458,1.04,0.981,-0.479,1.36,0.827,0,True
718,FUT8,75,0.647,0.0629,0.498,0.769,0.946,-0.353,1.65,0.867,1.03e-236,True
2138,TFRC,75,0.758,0.0621,0.499,0.658,0.954,-0.244,1.76,0.893,3.83e-287,True
2004,SLC5A3,75,0.619,0.0663,0.532,0.861,0.954,-0.451,1.69,0.893,8.75e-284,True
1637,PDIA4,75,0.615,0.0709,0.577,0.939,0.97,-0.544,1.77,0.88,0,True
888,GRN,75,0.761,0.0732,0.594,0.78,0.96,-0.431,1.95,0.907,0,True
1745,QSOX1,75,0.581,0.0743,0.597,1.03,0.952,-0.618,1.78,0.853,8.17e-273,True
319,CD82,75,0.543,0.0738,0.621,1.14,0.984,-0.703,1.79,0.773,0,True
256,CD151,75,0.712,0.0958,0.794,1.12,0.975,-0.883,2.31,0.76,0,True
471,COL6A3,75,0.966,0.0996,0.829,0.858,0.982,-0.698,2.63,0.88,0,True
1976,SLC39A6,75,0.79,0.104,0.864,1.09,0.981,-0.943,2.52,0.947,0,True
2227,TNFRSF25,75,0.462,0.106,0.894,1.94,0.987,-1.33,2.25,0.747,0,True
313,CD70,75,1.16,0.108,0.902,0.775,0.976,-0.647,2.97,0.92,0,True
1003,IL6R,75,0.442,0.108,0.903,2.04,0.982,-1.37,2.26,0.72,0,True
2268,TSPAN2,75,0.51,0.117,0.983,1.93,0.978,-1.46,2.48,0.747,0,True
1928,SLC1A4,75,0.608,0.131,1.1,1.81,0.989,-1.61,2.82,0.747,0,True
2228,TNFRSF4,75,0.203,0.138,1.18,5.81,0.994,-2.16,2.57,0.667,0,True
478,CPM,75,1.04,0.14,1.18,1.13,0.986,-1.33,3.41,0.747,0,True
1117,LGALS3,75,0.386,0.141,1.21,3.14,0.997,-2.04,2.82,0.587,0,True
325,CD9,75,1.19,0.182,1.57,1.32,0.987,-1.96,4.34,0.733,0,True
```

## §4a Most consistent antigens cohort-wide

**What:** genes with a pooled effect above 0.3 whose prediction interval excludes zero.

**How:** same model, ranked by mu.

```csv
index,gene,n_donors,mu,se,tau,tau_over_mu,I2,pi_lo,pi_hi,frac_donors_up,q_Q,on_shortlist
```

## §4b COPA, subgroup antigens

**What:** genes extreme in a minority of patients, which a mean test averages away.

**How:** median/MAD-standardised 75th percentile, sign-flip null, BH.

**Full table on disk:** `tables/tgt47_heterogeneity_copa.csv`

```csv
gene,copa_p75,frac_outlier,q_copa,mu,log2fc_paired,q,on_shortlist
GRIK2,86.7,0.187,0.0294,0.406,0,1.22e-05,False
OPCML,27.8,0.2,0.0294,0.367,0,8.1e-05,False
CD1B,22.7,0.173,0.292,0.00877,0,0.247,False
GPR37,21.3,0.187,0.453,0.013,0,0.303,False
GFRA3,16.6,0.173,0.242,0.0144,0,0.0977,False
LILRA6,13.3,0.173,0.437,0.0194,0,0.182,False
GALNT9,12.2,0.12,0.0294,0.115,0,0.00012,False
SORCS1,10.4,0.12,0.0294,0.37,0.0281,2.35e-06,False
LIPG,10.2,0.147,0.364,0.0326,0,0.154,False
GRIN2D,9.77,0.133,0.0452,0.0571,0,0.00244,False
PCDHGA11,9.29,0.133,0.0294,0.0531,0,0.000891,False
PTCHD1,8.53,0.187,0.0294,0.41,0.0235,4.9e-07,False
GLDN,8.16,0.2,0.216,0.0421,0,0.0179,False
SLC26A4,6.02,0.187,0.367,0.0159,0,0.242,False
SLC22A1,5.53,0.187,0.162,0.0736,0,0.0093,False
WNT5A,5.43,0.147,0.0294,0.143,0,0.000518,False
SUCNR1,5.32,0.173,0.368,0.00627,0,0.276,False
SMO,5.13,0.16,0.0852,0.0548,0,0.0203,False
THSD7A,5.1,0.187,0.154,0.0644,0,0.0237,False
PTGER1,5.04,0.187,0.0294,0.123,0,0.000134,False
CDH4,4.89,0.16,0.0294,0.451,0,3.44e-05,False
ENPEP,4.87,0.187,0.0294,0.112,0,0.005,False
IGLON5,4.73,0.24,1,0.111,0,0.517,False
KIR3DL2,4.6,0,0.0294,0.744,0,0.000947,False
FRAS1,4.5,0.2,0.711,0.123,0,0.182,False
CSPG5,4.43,0.16,0.122,0.025,0,0.00515,False
PSCA,4.37,0.12,0.186,0.00601,0,0.0402,False
NFAM1,4.33,0.173,0.311,0.0173,0,0.0357,False
PCDHGA5,4.3,0.147,0.0603,0.0923,0,0.00173,False
SLC1A3,4.2,0.227,0.196,0.0378,0,0.0331,False
SLC34A1,4.1,0.147,0.446,0.0147,0,0.164,False
IL17RD,4.1,0.187,0.154,0.0507,0,0.0125,False
CHRNB3,3.98,0.133,0.0294,0.0467,0,0.00251,False
ABCC3,3.94,0.2,0.649,-0.000132,0,0.645,False
PCDHGB7,3.86,0.213,0.133,0.0381,0,0.074,False
FAM171A2,3.82,0.173,0.0294,0.0852,0.00897,0.00133,False
CORIN,3.69,0.16,0.111,0.0598,0,0.0107,False
PCDHGA1,3.64,0.173,0.0294,0.0507,0,0.000145,False
TACR1,3.58,0.227,1,0.0193,0,0.592,False
SULF1,3.55,0.147,0.0747,0.127,0,0.0119,False
```

## §4b Bimodality of the per-donor effect

**What:** whether a gene's donor effects fall into two groups rather than one spread.

**How:** one- versus two-component Gaussian mixture BIC; delta_bic > 0 favours two components. A weak instrument at this donor count -- read it beside I2 and COPA.

**Full table on disk:** `tables/tgt47_heterogeneity_bimodality.csv`

```csv
gene,n_donors,delta_bic,two_component,mode_lo,mode_hi,w_hi,bimodality_coef,copa_p75,frac_outlier
CHGA,75,738,True,3.25e-05,0.444,0.12,0.92,0,0
NELL1,75,677,True,3.36e-14,0.569,0.214,0.897,0,0
KIAA0922,75,656,True,0,0.3,0.0802,0.855,0,0
HCRTR2,75,635,True,5.29e-05,0.281,0.161,0.886,0,0
MFI2,75,612,True,0,0.196,0.0669,0.872,0,0
MC5R,75,581,True,0.000104,0.19,0.148,0.935,0,0
DRD1,75,576,True,-4.95e-05,1.02,0.0133,0.972,0,0
MSLNL,75,575,True,7.94e-05,0.347,0.0133,0.973,0,0
PROKR2,75,569,True,4.1e-05,0.126,0.107,0.888,0,0
PCDH8,75,516,True,3.93e-05,0.214,0.0133,0.973,0,0
PCDHA10,75,509,True,-0.031,-4.55e-06,0.875,0.104,0,0
PCDHB4,75,499,True,0.000355,0.143,0.192,0.822,0,0
PON1,75,498,True,-3.23e-05,0.154,0.241,0.878,0,0
SLCO1B1,75,491,True,-0.514,0.000344,0.987,0.972,0,0
GLRA3,75,487,True,0.000126,0.125,0.135,0.829,0,0
CACNG4,75,477,True,9.92e-19,0.0769,0.188,0.888,0,0
IGHG2,75,476,True,1.23e-07,0.144,0.281,0.849,0,0
AJAP1,75,472,True,2.03e-05,0.107,0.264,0.618,0,0
HHIP,75,468,True,2.04e-10,0.291,0.361,0.823,0,0
NLGN1,75,466,True,1.25e-25,0.145,0.308,0.843,0,0
CACNG2,75,465,True,4.38e-05,0.0366,0.108,0.686,0,0
MMP16,75,462,True,6.26e-05,0.371,0.388,0.828,0,0
FCRL4,75,462,True,-2.44e-05,0.0555,0.24,0.848,0,0
GGT1,75,462,True,9.28e-06,0.291,0.373,0.866,0,0
GRM8,75,459,True,1.82e-05,0.157,0.321,0.886,0,0
```

## §4c Donor clusters against the clinical fields

**What:** whether the patients fall into groups on these antigens, and whether the groups mean anything clinically.

**How:** hierarchical clustering of the donor x gene effect matrix, silhouette cut, minimum 3 donors per cluster; chi-square against each categorical field.

**Full table on disk:** `tables/tgt47_donor_clusters.csv`

```csv
donor,dx2,study,sex,age,burden,lesion,lesion_class,clin_stage_class,is_cultured,cluster
B5__PT09,SS,buus2025,F,71,0.664,erythroderma,lesion,,False,1
B5__PT23,SS,buus2025,F,85,0.836,erythroderma,lesion,,False,1
D10__JONAK1,MF,jonak2021,F,33,0.153,plaque,lesion,early,False,1
D11__P1,MF,brentuximab2026,F,52,0.883,plaque,lesion,advanced,False,1
D11__P2,MF,brentuximab2026,F,48,0.899,tumor,lesion,advanced,False,1
D11__P3,MF,brentuximab2026,F,55,0.827,tumor,lesion,advanced,False,1
D11__P4,MF,brentuximab2026,F,45,0.254,plaque,lesion,advanced,False,1
D11__P5,MF,brentuximab2026,F,61,0.827,tumor,lesion,advanced,False,1
D11__P6,MF,brentuximab2026,M,64,0.769,plaque,lesion,advanced,False,2
D13__MF17,MF,gaydosik2022,F,76,0.985,tumor,lesion,advanced,False,1
D13__MF18,MF,gaydosik2022,M,87,0.469,tumor,lesion,advanced,False,2
D13__MF19,unresolved,gaydosik2022,M,62,0.518,tumor,lesion,advanced,False,1
D13__MF21,MF,gaydosik2022,M,61,0.177,tumor,lesion,advanced,False,2
D13__MF25,unresolved,gaydosik2022,M,68,0.354,erythroderma,lesion,advanced,False,1
D15__MF24,MF,il4ra2026,M,42,0.247,tumor,lesion,advanced,False,1
D15__MF26,MF,il4ra2026,M,30,0.92,erythroderma,lesion,advanced,False,1
D15__MF28,MF,il4ra2026,F,74,0.151,tumor,lesion,advanced,True,2
D15__MF30,MF,il4ra2026,M,80,0.584,erythroderma,lesion,advanced,True,2
D15__MF34,MF,il4ra2026,M,76,0.264,tumor,lesion,advanced,True,2
D15__MF35,MF,il4ra2026,F,66,0.786,mixed,lesion,advanced,True,2
D15__MF36,MF,il4ra2026,M,48,0.221,mixed,lesion,advanced,False,1
D15__MF48,MF,il4ra2026,M,71,0.837,papule,lesion,advanced,False,2
D1__P171,MF,chennareddy2025,M,84,0.932,tumor,lesion,advanced,False,1
D1__P204,MF,chennareddy2025,F,71,0.375,tumor,lesion,advanced,False,1
D1__P303,MF,chennareddy2025,F,65,0.899,tumor,lesion,advanced,False,2
D1__P312,MF,chennareddy2025,M,55,0.566,plaque,lesion,advanced,False,1
D1__PGS,MF,chennareddy2025,M,75,0.817,erythroderma,lesion,advanced,False,1
D3__P105,MF,brunner2024,M,84,0.185,erythroderma,lesion,advanced,False,1
D3__P192,SS,brunner2024,M,67,0.813,erythroderma,lesion,advanced,False,1
D3__P196,MF,brunner2024,F,75,0.791,erythroderma,lesion,advanced,False,1
D6__MF2,SS,gaydosik2019,F,68,0.489,erythroderma,lesion,advanced,False,1
D6__SC157,MF,gaydosik2019,F,83,0.572,tumor,lesion,advanced,False,2
D6__SC158,MF,gaydosik2019,F,46,0.756,plaque,lesion,advanced,False,1
D7__MF309,MF,rindler2021_mc,M,76,0.755,patch,lesion,advanced,False,2
D7__MF311,MF,rindler2021_mc,F,74,0.748,patch,lesion,advanced,False,1
D7__P73,MF,rindler2021_mc,M,82,0.327,plaque,lesion,advanced,False,2
D7__P84,MF,rindler2021_mc,M,75,0.327,plaque,lesion,advanced,False,1
D8__P138,MF,alkon2024,M,47,0.505,patch,lesion,early,False,2
D8__P150,MF,alkon2024,M,56,0.724,patch,lesion,early,False,2
D8__P163,MF,alkon2024,F,83,0.679,patch,lesion,early,False,2
D8__P65,MF,alkon2024,M,53,0.496,patch,lesion,early,False,2
D8__P90,MF,alkon2024,M,75,0.227,patch,lesion,early,False,2
H__MFIV1,MF,herrera2021,F,39,0.62,plaque,lesion,advanced,False,1
H__SS1,SS,herrera2021,F,49,0.33,patch,lesion,advanced,False,1
H__SS4,SS,herrera2021,M,56,0.816,erythroderma,lesion,advanced,False,1
Li2024_atlas__CTCL1,MF,li2024,F,38,0.956,mixed,lesion,advanced,False,1
Li2024_atlas__CTCL10,MF,li2024,F,56,0.88,skin_unspecified,unspecified,early,False,2
Li2024_atlas__CTCL12,MF,li2024,M,60,0.894,skin_unspecified,unspecified,advanced,False,1
Li2024_atlas__CTCL13,MF,li2024,M,68,0.674,skin_unspecified,unspecified,advanced,False,1
Li2024_atlas__CTCL14,MF,li2024,F,68,0.288,skin_unspecified,unspecified,early,False,2
Li2024_atlas__CTCL16,MF,li2024,F,37,0.779,skin_unspecified,unspecified,early,False,1
Li2024_atlas__CTCL17,MF,li2024,M,76,0.712,skin_unspecified,unspecified,advanced,False,2
Li2024_atlas__CTCL18,MF,li2024,M,68,0.982,skin_unspecified,unspecified,advanced,False,2
Li2024_atlas__CTCL2,MF,li2024,M,67,0.382,mixed,lesion,early,False,2
Li2024_atlas__CTCL3,MF,li2024,F,74,0.832,mixed,lesion,early,False,2
Li2024_atlas__CTCL4,MF,li2024,M,57,0.136,mixed,lesion,early,False,2
Li2024_atlas__CTCL5,MF,li2024,M,56,0.649,mixed,lesion,advanced,False,2
Li2024_atlas__CTCL6,MF,li2024,M,67,0.277,mixed,lesion,early,False,2
Li2024_atlas__CTCL7,MF,li2024,F,59,0.0869,mixed,lesion,early,False,2
Li2024_atlas__CTCL8,MF,li2024,M,61,0.751,mixed,lesion,early,False,2
Li2024_atlas__CTCL9,MF,li2024,F,46,0.954,skin_unspecified,unspecified,early,False,2
Li2024_atlas__MF17,MF,li2024,M,36,0.00837,tumor,lesion,advanced,False,2
Li2024_atlas__MF21,MF,li2024,M,79,0.713,tumor,lesion,advanced,False,1
Li2024_atlas__MF22,MF,li2024,M,71,0.343,plaque,lesion,early,False,2
Li2024_atlas__MF27,MF,li2024,F,57,0.129,tumor,lesion,advanced,False,2
Li2024_atlas__MF28,MF,li2024,M,60,0.845,tumor,lesion,advanced,False,1
Li2024_atlas__MF30,MF,li2024,F,48,0.549,tumor,lesion,advanced,False,1
Li2024_atlas__PT11,MF,li2024,M,50,0.874,patch,lesion,advanced,False,2
Li2024_atlas__PT35,MF,li2024,M,61,0.919,mixed,lesion,advanced,False,1
Li2024_atlas__PT47,MF,li2024,M,60,0.65,mixed,lesion,advanced,False,2
Li2024_atlas__PT50,MF,li2024,M,73,0.441,mixed,lesion,early,False,1
Li2024_atlas__PT52,MF,li2024,F,55,0.643,mixed,lesion,advanced,False,1
Li2024_atlas__PT53,MF,li2024,M,58,0.658,mixed,lesion,advanced,False,2
Li2024_atlas__PT55,MF,li2024,M,71,0.968,mixed,lesion,advanced,False,2
Li2024_atlas__PT56,MF,li2024,M,65,0.343,mixed,lesion,advanced,False,2
```

## §4c Per-cluster mean effect for the leading antigens

**What:** what separates the donor clusters, where there are any.

**How:** mean per-donor effect inside each cluster; `spread` is max minus min across clusters.

**Full table on disk:** `tables/tgt47_donor_cluster_profiles.csv`

```csv
gene,1,2,spread
LGALS3,-0.559,1.33,1.89
APP,1.29,-0.559,1.85
NTRK2,-0.00774,1.5,1.51
FGFR1,1.11,-0.301,1.41
CPM,1.69,0.349,1.34
ADAM12,-0.199,1.05,1.25
SLC1A4,-0.00337,1.22,1.22
KIR3DL2,1.32,0.149,1.17
CHRNA6,1.2,0.0411,1.16
SGCE,1.35,0.261,1.09
TNFRSF25,0.991,-0.0834,1.07
IL9R,-0.0152,1.05,1.06
SLC1A5,-0.101,0.896,0.997
ALCAM,0.884,-0.109,0.992
SEMA4A,-0.144,0.8,0.944
EPHB6,0.875,-0.0501,0.925
PTH2R,1.43,0.556,0.873
ADGRE1,1.15,0.289,0.86
CD9,0.767,1.62,0.853
TSPAN13,0.0255,0.877,0.852
TIMP1,-0.113,0.722,0.836
IL6R,0.847,0.0311,0.816
TNFRSF4,0.576,-0.195,0.771
MME,0.801,0.059,0.742
IGHM,0.827,0.148,0.679
```

## §4e Eligibility

**What:** the fraction of patients in whom at least X% of malignant CD4 cells carry the antigen.

**How:** transcript detection per donor, thresholded at 5/10/25/50/75%.

**Full table on disk:** `tables/tgt47_F2_eligibility_source.csv`

```csv
index,gene,n_donors,median_pct_expr,frac_donors_ge5,frac_donors_ge10,frac_donors_ge25,frac_donors_ge50,frac_donors_ge75,mu,I2
27,SURF4,75,0.439,1,0.987,0.907,0.373,0.12,0.449,0.975
67,CD82,75,0.495,1,1,0.853,0.493,0.0667,0.543,0.984
83,TMEM30A,75,0.193,1,0.88,0.347,0.0133,0,0.433,0.91
103,VDAC1,75,0.494,1,0.987,0.88,0.493,0.0933,0.648,0.956
137,SYPL1,75,0.4,0.987,0.96,0.693,0.253,0,0.617,0.961
151,TFRC,75,0.294,0.987,0.96,0.6,0.2,0.04,0.758,0.954
162,GRN,75,0.212,0.987,0.867,0.453,0.107,0,0.761,0.96
165,AP2S1,75,0.37,0.987,0.933,0.733,0.28,0.0133,0.57,0.958
168,GPI,75,0.482,0.987,0.973,0.827,0.453,0.08,0.44,0.981
187,PDIA4,75,0.261,0.973,0.893,0.533,0.133,0.0133,0.615,0.97
202,SLC39A6,75,0.187,0.973,0.92,0.293,0.0667,0,0.79,0.981
218,CALU,75,0.193,0.96,0.867,0.347,0.0533,0,0.614,0.92
221,LGALS3,75,0.48,0.96,0.88,0.773,0.48,0.253,0.386,0.997
276,SLC5A3,75,0.209,0.92,0.773,0.453,0.0933,0,0.619,0.954
277,QSOX1,75,0.138,0.92,0.693,0.213,0.0267,0,0.581,0.952
285,CD70,75,0.215,0.92,0.813,0.427,0.0533,0,1.16,0.976
290,TNFRSF25,75,0.328,0.92,0.88,0.613,0.28,0.04,0.462,0.987
295,TNFRSF4,75,0.39,0.907,0.893,0.6,0.253,0.0533,0.203,0.994
299,STX6,75,0.143,0.907,0.68,0.187,0.0133,0,0.499,0.84
300,FUT8,75,0.183,0.907,0.747,0.4,0.12,0.0267,0.647,0.946
312,CD151,75,0.2,0.893,0.827,0.427,0.107,0.0133,0.712,0.975
333,IL6R,75,0.234,0.867,0.773,0.467,0.0667,0,0.442,0.982
348,CD9,75,0.262,0.84,0.733,0.507,0.227,0.0267,1.19,0.987
373,SLC4A2,75,0.119,0.813,0.64,0.2,0.0267,0,0.796,0.906
377,SLC1A4,75,0.143,0.813,0.627,0.307,0.0533,0.0133,0.608,0.989
397,TSPAN2,75,0.133,0.787,0.653,0.187,0.0133,0,0.51,0.978
455,COL6A3,75,0.111,0.693,0.56,0.173,0.107,0.0267,0.966,0.982
469,CPM,75,0.133,0.667,0.573,0.24,0.0533,0.0133,1.04,0.986
```

## §4f Large and consistent vs large and selective

**What:** the two quadrants a development decision turns on.

**How:** mu > 0.5 split on whether tau is below the cohort median and the prediction interval excludes zero.

```csv
index,gene,quadrant,mu,tau,pi_lo,pi_hi,frac_donors_up,on_shortlist
87,SLC1A4,large_but_variable,0.608,1.1,-1.61,2.82,0.747,True
98,IL9R,large_but_variable,0.53,1.2,-1.89,2.95,0.64,False
126,SGCE,large_but_variable,0.811,1.02,-1.24,2.86,0.627,False
375,KIR3DL2,large_but_variable,0.744,1.51,-2.29,3.78,0.36,False
588,PLD1,large_but_variable,0.783,0.933,-1.09,2.66,0.747,False
663,PRSS21,large_but_variable,0.872,1.14,-1.42,3.16,0.587,False
692,NTRK2,large_but_variable,0.737,1.39,-2.05,3.52,0.6,False
694,NRN1,large_but_variable,0.867,0.866,-0.871,2.61,0.773,False
814,PTH2R,large_but_variable,0.999,1.23,-1.46,3.46,0.667,False
933,CHRNA6,large_but_variable,0.627,0.874,-1.13,2.38,0.68,False
978,CD9,large_but_variable,1.19,1.57,-1.96,4.34,0.733,True
1014,THY1,large_but_variable,0.649,1.13,-1.62,2.91,0.587,False
1102,ADGRE1,large_but_variable,0.725,0.926,-1.13,2.58,0.707,False
1132,AEBP1,large_but_variable,0.533,0.87,-1.21,2.28,0.64,False
1252,TSPAN3,large_but_variable,0.539,0.552,-0.568,1.65,0.84,False
1342,FGFR2,large_but_variable,0.528,0.852,-1.18,2.24,0.6,False
1441,GPR55,large_but_variable,0.568,0.969,-1.38,2.51,0.653,False
1636,FAM174B,large_but_variable,0.604,0.79,-0.982,2.19,0.747,False
1649,EVC2,large_but_variable,0.63,0.628,-0.631,1.89,0.853,False
1703,CPM,large_but_variable,1.04,1.18,-1.33,3.41,0.747,True
1704,MRC2,large_but_variable,1.07,1.04,-1.01,3.15,0.88,False
1711,CD82,large_but_variable,0.543,0.621,-0.703,1.79,0.773,True
1731,CRELD2,large_but_variable,0.502,0.87,-1.24,2.25,0.827,False
1733,COL6A3,large_but_variable,0.966,0.829,-0.698,2.63,0.88,True
1737,GALNT2,large_but_variable,0.565,0.989,-1.42,2.55,0.72,False
1742,SLC39A6,large_but_variable,0.79,0.864,-0.943,2.52,0.947,True
1763,TSPAN2,large_but_variable,0.51,0.983,-1.46,2.48,0.747,True
1773,TNFSF4,large_but_variable,0.555,0.87,-1.19,2.3,0.68,False
1777,CD70,large_but_variable,1.16,0.902,-0.647,2.97,0.92,True
1782,CD151,large_but_variable,0.712,0.794,-0.883,2.31,0.76,True
1816,PDIA4,large_but_variable,0.615,0.577,-0.544,1.77,0.88,True
1825,SLC39A14,large_but_variable,0.658,0.736,-0.819,2.13,0.853,False
1862,QSOX2,large_but_variable,0.653,0.715,-0.783,2.09,0.853,False
1878,SYPL1,large_but_variable,0.617,0.446,-0.279,1.51,0.907,True
1879,GRN,large_but_variable,0.761,0.594,-0.431,1.95,0.907,True
1892,AP2S1,large_but_variable,0.57,0.423,-0.279,1.42,0.907,True
1902,VDAC1,large_but_variable,0.648,0.341,-0.0378,1.33,0.947,True
1912,TFRC,large_but_variable,0.758,0.499,-0.244,1.76,0.893,True
1913,SLC5A3,large_but_variable,0.619,0.532,-0.451,1.69,0.893,True
1923,QSOX1,large_but_variable,0.581,0.597,-0.618,1.78,0.853,True
1938,FUT8,large_but_variable,0.647,0.498,-0.353,1.65,0.867,True
1943,SLC29A1,large_but_variable,0.861,0.528,-0.199,1.92,0.933,False
1976,SDC4,large_but_variable,0.535,0.569,-0.608,1.68,0.747,False
2034,CALU,large_but_variable,0.614,0.423,-0.236,1.46,0.88,True
2073,SLC4A2,large_but_variable,0.796,0.443,-0.0948,1.69,0.933,True
2079,SGCB,large_but_variable,0.526,0.436,-0.351,1.4,0.867,False
```

======================== COPY TO HERE ==========================