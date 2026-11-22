> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 46 — Blood, pooled and stratified

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**, pooled rather than paired. Notebook 43 compares malignant and
reactive CD4 cells WITHIN each donor; this one pools all malignant CD4 cells against all reactive
CD4 cells, which lets donors with only one arm contribute and makes clinical strata comparable as
cell populations.

Cohort: 75 independent donor keys (52 with both arms, 11 with one arm,
12 healthy). The geskin2026 ex vivo drug wells and duplicate persons were removed in 43
section 0b. Malignancy call `mal_cluster`.

Weighting: donor-balanced is the headline (every donor contributes equal weight to its arm);
cell-weighted is printed beside it as `*_cells`. This matters here -- three harro2023 early-MF
donors with no malignant arm hold half of all reactive cells under cell weighting.

Effect sizes: `log2fc` (log2 ratio of pooled mean log1p CP10K, pseudocount 0.05), `d_det`
(difference in the fraction of cells detected), `cohen_d`, and `auroc` (probability a random
malignant cell exceeds a random comparator cell; 0.5 = no separation, capped by detection for
sparse genes). CIs from a 1000-draw donor bootstrap. The headline p is van Elteren
(`q_blocked`): donor-level means ranked within study. No p-value is computed over cells.

Stratified contrasts, per field: (a) malignant vs reactive within each stratum, (b) malignant
cells of one stratum vs the other, (c) the same for reactive cells, (d) difference-in-differences.
p-values permute donor labels within study (50,000 permutations, `q_study`), with a free
permutation beside them (`q_free`). Fields are tier 1 (tested), 2 (described) or 3 (counted).

Everything is transcript, in a 10,000-gene HVG subset. Pooling adds cells, not patients: the
metadata gaps of notebook 44 are unchanged.

## §0b Cohort by arm pattern

**What:** how many independent donors have both CD4 arms, one, or are healthy.

**How:** an arm counts when it holds >= 20 cells.

**Full table on disk:** `tables/tgt46_cohort.csv`

```csv
pattern,HC,MF,MF/SS,MF_or_SS_unresolved,SS,All
healthy,12,0,0,0,0,12
malignant only,0,0,0,0,6,6
paired,0,8,2,5,37,52
reactive only,0,3,0,1,1,5
All,12,11,2,6,44,75
```

## §1 Pool composition

**What:** who each pool is actually made of, under each weighting.

**How:** Kish effective donors (sum(w)^2/sum(w^2)).

**Full table on disk:** `tables/tgt46_pool_composition.csv`

```csv
arm,n_donors,n_cells,kish_n_eff_cells,kish_n_eff_donor,top_donor,top_donor_share_cells,top_study_share_cells,top_study
CD4_malignant,58,295656,24.2,58,B2__CTCL2,0.103,0.273,ren2023
CD4_reactive,57,111125,7.75,57,B7__MF1,0.297,0.54,harro2023
CD4_HC,12,40400,6.02,12,B6__N2,0.245,0.672,ren2023
```

## §2a Pooled shortlist

**What:** the genes passing every pooled gate: clean in 43, log2FC > 0.5 with the bootstrap CI above zero, van Elteren q < 0.05, not a loss marker or lineage control.

**How:** donor-balanced pooled contrast, malignant vs reactive CD4.

**Full table on disk:** `tables/tgt46_pooled_all_genes.csv`

```csv
index,gene,cd_name,log2fc,log2fc_lo,log2fc_hi,pct_a,pct_b,d_det,auroc,auroc_lo,cohen_d,log2fc_cells,log2fc_paired,q_blocked,log2fc_vs_hc,auroc_vs_hc,on_43_shortlist
570,KIR3DL2,CD158k,3.04,2.78,3.3,0.396,0.0214,0.374,0.688,0.653,0.977,3.37,2.79,1.04e-09,3.46,0.695,True
1048,TPO,,2.76,2.38,3.1,0.236,0.00888,0.227,0.614,0.583,0.702,2.96,1.85,1.81e-12,2.98,0.617,True
827,RAB25,,1.98,1.53,2.36,0.142,0.00782,0.135,0.567,0.543,0.506,1.79,0.375,0.000295,2,0.568,False
430,GPR15,,1.94,1.58,2.3,0.205,0.0338,0.172,0.587,0.562,0.535,2.08,1.44,1.28e-05,2.44,0.597,True
168,CD70,CD70,1.94,1.68,2.22,0.251,0.0443,0.206,0.604,0.584,0.583,2.29,1.93,2.71e-09,2.38,0.615,True
362,FCRL3,,1.91,1.59,2.24,0.214,0.0353,0.179,0.59,0.568,0.539,2.01,1.54,4.03e-06,2.27,0.598,True
814,PTH2R,,1.74,1.33,2.13,0.0908,0.00311,0.0877,0.544,0.53,0.386,1.29,0.971,4.48e-08,1.82,0.545,False
982,STAM,,1.64,1.39,1.86,0.396,0.124,0.272,0.642,0.612,0.665,1.8,1.64,1.25e-07,1.85,0.652,True
249,CPM,,1.5,1.14,1.84,0.103,0.0144,0.0887,0.544,0.53,0.371,1.33,1.1,5.19e-09,1.61,0.547,False
1064,TSPAN2,,1.47,1.19,1.75,0.214,0.058,0.156,0.579,0.56,0.458,1.69,0.938,9.26e-09,1.96,0.593,True
1009,TIGIT,,1.43,1.15,1.76,0.437,0.147,0.29,0.647,0.621,0.633,1.56,1.65,4.05e-09,1.46,0.649,True
323,EPCAM,,1.38,0.815,1.82,0.0647,0.00126,0.0635,0.532,0.515,0.344,0.982,0.0183,0.0014,1.4,0.532,False
773,PI16,,1.35,1.01,1.69,0.29,0.105,0.185,0.595,0.569,0.487,1.61,1.17,2.58e-06,1.92,0.617,True
368,FGFR1,,1.32,1.08,1.53,0.11,0.0225,0.0872,0.544,0.533,0.347,1.29,1.15,5.19e-09,1.55,0.549,False
703,NRROS,,1.28,1.08,1.46,0.192,0.0612,0.131,0.566,0.551,0.398,1.3,1.2,1.63e-08,1.35,0.569,True
326,EPHA4,,1.19,1,1.38,0.217,0.0761,0.141,0.571,0.557,0.388,1.42,0.989,5.24e-11,1.53,0.583,True
176,CD84,CD84 / SLAMF5,1.17,1.01,1.33,0.284,0.118,0.166,0.586,0.57,0.434,1.3,1.07,8.24e-12,1.43,0.599,True
979,ST6GALNAC1,,1.15,0.81,1.46,0.119,0.0367,0.082,0.542,0.525,0.319,1.34,0.361,0.00562,1.42,0.549,False
184,CDH1,,1.14,0.822,1.4,0.055,0.00276,0.0522,0.526,0.517,0.295,1.02,0.343,4.05e-05,0.993,0.525,False
11,ACVR2A,,1.13,0.914,1.37,0.115,0.0322,0.0824,0.541,0.53,0.302,1.16,0.886,6.22e-09,1.31,0.545,False
569,KIR3DL1,,1.13,0.681,1.51,0.056,0.00478,0.0512,0.526,0.512,0.282,1.36,0.0811,0.0179,1.03,0.525,False
552,ITGB1,,1.11,0.92,1.31,0.791,0.451,0.34,0.739,0.696,0.936,1.17,1.16,6.03e-11,1.35,0.769,True
24,ADGRE1,,1.1,0.825,1.37,0.114,0.0325,0.0814,0.541,0.53,0.307,1.01,1.15,1.06e-07,1.24,0.545,False
443,GPR55,,1.09,0.831,1.37,0.0842,0.018,0.0662,0.533,0.525,0.284,1.12,0.968,2.28e-09,1.18,0.535,False
761,PDCD1,PD-1,1.07,0.731,1.41,0.148,0.0506,0.0976,0.549,0.532,0.314,1.44,1.22,1.49e-09,1.58,0.562,True
429,GPR146,,1.05,0.812,1.27,0.144,0.0503,0.0935,0.547,0.534,0.314,1.24,0.824,5.35e-07,1.05,0.548,True
584,LAG3,,1.04,0.685,1.36,0.126,0.0443,0.0819,0.541,0.525,0.288,1.4,0.841,0.00238,0.782,0.537,False
111,CCR4,CCR4,1.03,0.709,1.36,0.243,0.112,0.131,0.567,0.543,0.36,1.21,1.13,0.000815,1.47,0.587,True
700,NRN1,,1.03,0.745,1.29,0.0496,0.00227,0.0473,0.524,0.515,0.284,0.998,0.447,2.87e-05,1.07,0.524,False
710,NXPH4,,1.02,0.685,1.34,0.0462,0.00268,0.0435,0.522,0.513,0.27,1.15,0.347,5.2e-07,1.07,0.522,False
4,ABCC1,,0.949,0.756,1.13,0.255,0.123,0.131,0.567,0.552,0.343,1.06,0.907,8.75e-07,1.21,0.582,True
902,SLAMF6,,0.928,0.777,1.09,0.278,0.138,0.141,0.572,0.558,0.359,1.18,0.771,3.73e-08,1.16,0.586,True
388,FZD3,,0.924,0.548,1.29,0.0588,0.0157,0.0431,0.522,0.511,0.233,1.04,0.225,0.00813,0.953,0.522,False
669,MRC1,,0.921,0.539,1.3,0.0361,0.00258,0.0335,0.517,0.508,0.232,0.871,0.161,0.000694,0.685,0.513,False
334,EVC2,,0.884,0.73,1.03,0.0584,0.0128,0.0456,0.523,0.518,0.238,0.858,0.841,1.33e-12,1.09,0.527,False
431,GPR155,,0.845,0.627,1.1,0.209,0.107,0.102,0.552,0.537,0.289,0.925,0.576,7.13e-06,0.52,0.537,False
509,IGHM,,0.841,0.617,1.1,0.0454,0.00603,0.0393,0.52,0.514,0.235,0.808,0.488,9.73e-06,0.214,0.508,False
265,CTLA4,,0.834,0.508,1.2,0.209,0.11,0.0996,0.551,0.53,0.28,0.906,0.623,0.000344,0.951,0.558,True
17,ADAM29,,0.828,0.418,1.22,0.0294,0.000575,0.0289,0.514,0.507,0.225,0.836,0.0332,8.37e-05,0.839,0.515,False
643,MBP,,0.813,0.698,0.93,0.696,0.475,0.221,0.669,0.64,0.637,0.768,0.843,2.14e-10,0.995,0.695,True
204,CHRNA7,,0.796,0.506,1.11,0.0292,0.000747,0.0284,0.514,0.508,0.222,0.739,0.0981,1.09e-07,0.817,0.515,False
174,CD82,,0.788,0.619,0.951,0.416,0.252,0.164,0.591,0.569,0.394,0.717,0.765,1.68e-06,0.998,0.61,True
386,FUT8,,0.78,0.581,0.967,0.242,0.135,0.107,0.555,0.54,0.283,0.874,0.793,1.71e-07,0.841,0.557,True
999,TFRC,,0.773,0.619,0.923,0.141,0.0672,0.0741,0.537,0.529,0.234,0.781,0.746,2.24e-08,0.997,0.545,True
370,FGL2,,0.768,0.549,1.01,0.0628,0.0214,0.0413,0.521,0.514,0.199,1.24,0.55,0.000333,0.463,0.513,False
100,CADM1,,0.753,0.409,1.1,0.0495,0.0139,0.0355,0.518,0.508,0.194,0.561,0.268,0.00127,0.927,0.521,False
536,IL6R,,0.747,0.592,0.901,0.298,0.177,0.121,0.564,0.548,0.31,0.555,0.738,2.66e-08,0.857,0.574,True
261,CSF2RB,,0.737,0.508,0.963,0.0409,0.00754,0.0334,0.517,0.511,0.201,0.855,0.444,3.05e-05,0.774,0.517,False
329,EPHB6,,0.713,0.406,1.02,0.151,0.0788,0.0721,0.536,0.52,0.219,0.942,0.231,0.0192,0.887,0.544,False
396,GALNT9,,0.691,0.386,0.994,0.0265,0.000896,0.0256,0.513,0.506,0.208,0.692,0.0504,0.000665,0.717,0.513,False
342,F5,,0.68,0.432,0.924,0.117,0.0596,0.0571,0.529,0.517,0.196,0.83,0.684,0.00367,1.21,0.544,False
144,CD28,,0.674,0.526,0.818,0.409,0.272,0.137,0.578,0.559,0.341,0.792,0.605,4.48e-08,0.826,0.593,True
322,ENTPD1,,0.673,0.36,0.969,0.0742,0.0314,0.0427,0.521,0.511,0.178,0.389,0.404,0.00721,0.995,0.528,False
50,ANXA2,,0.662,0.464,0.89,0.531,0.35,0.181,0.603,0.571,0.399,1.01,0.706,9.94e-05,0.985,0.638,True
171,CD79B,,0.662,0.411,0.898,0.165,0.0909,0.0736,0.537,0.521,0.218,0.694,0.548,6.42e-05,0.703,0.539,False
457,GRN,,0.654,0.501,0.818,0.167,0.094,0.0731,0.537,0.527,0.219,0.793,0.624,2.99e-06,0.441,0.527,True
901,SLAMF1,,0.639,0.403,0.874,0.241,0.15,0.0914,0.547,0.528,0.241,0.594,0.578,0.00169,0.477,0.54,False
401,GDPD5,,0.635,0.422,0.819,0.095,0.0474,0.0476,0.524,0.515,0.179,0.668,0.628,0.000194,0.912,0.531,False
583,L1CAM,,0.631,0.419,0.847,0.0309,0.00412,0.0268,0.513,0.508,0.195,0.373,0.201,2.46e-07,0.576,0.513,False
108,CCR10,,0.628,0.178,1.05,0.0753,0.0377,0.0377,0.519,0.505,0.152,0.665,0.144,0.0269,0.998,0.526,False
385,FUT7,,0.616,0.345,0.903,0.112,0.0554,0.0565,0.528,0.515,0.18,0.605,0.323,0.00765,0.726,0.532,False
512,IGSF8,,0.616,0.491,0.749,0.214,0.129,0.0857,0.543,0.532,0.226,0.612,0.536,6.69e-07,0.382,0.53,False
899,SIRPG,,0.608,0.439,0.789,0.27,0.179,0.091,0.549,0.532,0.248,0.468,0.633,7.08e-05,0.437,0.537,True
567,KIR2DL1,,0.595,0.202,0.993,0.0226,0.00297,0.0197,0.51,0.503,0.173,1.14,0.0143,0.0208,0.52,0.509,False
499,ICAM2,,0.592,0.424,0.744,0.508,0.364,0.144,0.589,0.562,0.358,0.501,0.58,7.9e-05,0.442,0.568,False
213,CLDND1,,0.59,0.424,0.763,0.478,0.343,0.135,0.581,0.555,0.332,0.667,0.413,0.000319,0.687,0.592,False
811,PTGER4,,0.572,0.369,0.751,0.345,0.242,0.103,0.557,0.535,0.262,0.86,0.556,0.0036,0.966,0.586,False
911,SLC1A1,,0.571,0.331,0.811,0.0224,0.00288,0.0195,0.51,0.505,0.164,0.425,0.0661,0.000935,0.644,0.511,False
944,SLC5A3,,0.57,0.384,0.758,0.127,0.0739,0.053,0.526,0.517,0.168,0.755,0.535,0.000134,0.64,0.53,False
593,LAYN,,0.548,0.31,0.772,0.036,0.0129,0.0232,0.512,0.506,0.146,0.563,0.302,0.0123,0.642,0.513,False
358,FCGRT,,0.543,0.308,0.744,0.156,0.0994,0.0566,0.529,0.514,0.179,0.436,0.451,0.0273,0.353,0.521,False
675,MUC1,,0.535,0.349,0.72,0.0281,0.00587,0.0223,0.511,0.507,0.167,0.367,0.23,4.12e-05,0.545,0.512,False
503,IFNGR2,,0.532,0.316,0.757,0.204,0.132,0.0729,0.537,0.521,0.192,0.634,0.497,0.00161,0.132,0.51,False
371,FLNA,,0.527,0.387,0.666,0.633,0.48,0.153,0.605,0.577,0.386,0.549,0.458,4.38e-06,0.995,0.669,False
915,SLC1A5,,0.523,0.338,0.714,0.0985,0.0547,0.0438,0.522,0.513,0.153,0.573,0.46,0.00127,0.284,0.514,False
369,FGFR2,,0.51,0.337,0.705,0.018,0.00176,0.0163,0.508,0.505,0.159,0.269,0.177,0.000152,0.465,0.507,False
1031,TMX4,,0.502,0.364,0.647,0.324,0.224,0.0998,0.552,0.536,0.226,0.463,0.403,6.03e-05,0.592,0.559,False
1028,TMEM245,,0.501,0.317,0.709,0.179,0.124,0.0545,0.528,0.517,0.169,0.476,0.474,0.00693,0.591,0.533,False
```

## §2a Controls

**What:** the antigens with a known answer, pooled.

**How:** CCR4, KIR3DL2 and IL2RA should be up; CD7 and DPP4 down with AUROC below 0.5.

```csv
gene,log2fc,log2fc_lo,log2fc_hi,d_det,auroc,auroc_lo,auroc_hi,log2fc_paired,q_blocked
CCR4,1.03,0.709,1.36,0.131,0.567,0.543,0.594,1.13,0.000815
KIR3DL2,3.04,2.78,3.3,0.374,0.688,0.653,0.723,2.79,1.04e-09
IL2RA,-0.215,-0.577,0.122,-0.0125,0.493,0.48,0.507,-0.19,0.138
CD7,-1.55,-1.94,-1.18,-0.422,0.276,0.242,0.313,-1.99,6.03e-11
DPP4,-1.85,-2.02,-1.7,-0.134,0.433,0.423,0.443,-1.91,2.26e-13
CD5,0.000917,-0.171,0.184,-0.0108,0.498,0.472,0.528,-0.0125,0.939
CD6,-0.536,-0.749,-0.351,-0.147,0.416,0.39,0.442,-0.459,8.11e-06
CD27,0.558,0.429,0.69,0.135,0.61,0.581,0.644,0.528,4.87e-06
CD2,-0.641,-0.872,-0.432,-0.203,0.381,0.348,0.414,-0.504,1.41e-06
```

## §2b Pooled vs paired agreement

**What:** whether pooling reproduces 43's paired ranking.

**How:** Spearman over all genes.

```csv
key,value
spearman_pooled_vs_paired,0.834
spearman_donor_balanced_vs_cell_weighted,0.887
```

## §2c Leave-one-study-out

**What:** whether any single study carries the pooled effect.

**How:** the pooled log2FC recomputed with each study dropped in turn.

**Full table on disk:** `tables/tgt46_loso.csv`

```csv
gene,pooled,loso_min,loso_max,study_moving_most,sign_kept_every_study
KIR3DL2,3.04,2.99,3.17,geskin2026,True
TPO,2.76,2.61,2.87,borcherding2023,True
RAB25,1.98,1.74,2.09,buus2025,True
GPR15,1.94,1.68,2.01,buus2025,True
CD70,1.94,1.88,2.02,geskin2026,True
FCRL3,1.91,1.74,2.06,buus2025,True
PTH2R,1.74,1.07,1.84,buus2025,True
STAM,1.64,1.57,1.69,buus2025,True
CPM,1.5,1.38,1.61,buus2025,True
TSPAN2,1.47,1.35,1.63,buus2025,True
TIGIT,1.43,1.36,1.61,geskin2026,True
EPCAM,1.38,1.16,1.47,borcherding2023,True
PI16,1.35,1.13,1.45,ren2023,True
FGFR1,1.32,1.18,1.37,ren2023,True
NRROS,1.28,1.22,1.33,ren2023,True
CCR4,1.03,0.991,1.08,rindler2021_fi,True
IL2RA,-0.215,-0.345,-0.0985,buus2025,True
CD7,-1.55,-1.61,-1.48,gaydosik2022,True
DPP4,-1.85,-1.89,-1.83,harro2023,True
```

## §3 Power table

**What:** donors per arm on each side of every split, and the resulting tier.

**How:** tier 1 = tested, 2 = described, 3 = counted only.

**Full table on disk:** `tables/tgt46_power_table.csv`

```csv
field,x,y,n_mal_x,n_mal_y,tier_mal,n_rea_x,n_rea_y,tier_rea,tier_did
disease,MF,SS,8,43,1,11,38,1,1
stage IV vs I-III,IV,I-III,33,7,1,30,11,1,1
stage class,advanced,early,38,2,3,36,5,1,3
B stage,B2,B0-B1,15,12,1,13,15,1,1
ECP,yes,no,27,23,1,26,22,1,1
retinoid,yes,no,16,34,1,14,34,1,1
interferon,yes,no,8,42,1,7,41,1,1
HDACi,yes,no,7,43,1,6,42,1,1
treatment-naive,yes,no,4,45,2,4,42,2,2
sex,F,M,24,34,1,24,33,1,1
age (median 70),older,younger,29,20,1,28,21,1,1
burden (median 0.90),high,low,32,26,1,26,31,1,1
blood tumour % (median 59),high,low,11,10,1,10,11,1,1
cultured 2 days ex vivo,yes,no,6,52,1,5,52,1,1
```

## §4a Stratified summary

**What:** how many genes differ between the levels of each field, for tumour cells, reactive cells and the difference-in-differences.

**How:** within-study donor permutation (q_study) and free permutation (q_free), BH.

**Full table on disk:** `tables/tgt46_strat_summary.csv`

```csv
field,x vs y,mal n,rea n,tier mal/rea/did,log10 cfg study,tumour q_study,tumour q_free,reactive q_study,reactive q_free,DiD q_study,DiD q_free
disease,MF vs SS,8/43,11/38,1/1/1,5.43,0,0,0,0,0,0
stage IV vs I-III,IV vs I-III,33/7,30/11,1/1/1,4.38,0,0,0,0,1,9
stage class,advanced vs early,38/2,36/5,3/1/3,,,,0,0,,
B stage,B2 vs B0-B1,15/12,13/15,1/1/1,5.1,0,0,0,0,0,0
ECP,yes vs no,27/23,26/22,1/1/1,6.87,0,0,0,2,0,2
retinoid,yes vs no,16/34,14/34,1/1/1,5.2,0,2,0,0,0,0
interferon,yes vs no,8/42,7/41,1/1/1,6.07,0,0,0,0,0,0
HDACi,yes vs no,7/43,6/42,1/1/1,4.95,0,0,0,0,0,0
treatment-naive,yes vs no,4/45,4/42,2/2/2,,,,,,,
sex,F vs M,24/34,24/33,1/1/1,11.2,0,0,0,0,0,0
age (median 70),older vs younger,29/20,28/21,1/1/1,8,0,1,0,0,0,0
burden (median 0.90),high vs low,32/26,26/31,1/1/1,9.12,0,0,8,9,0,0
blood tumour % (median 59),high vs low,11/10,10/11,1/1/1,4.44,0,0,0,0,0,0
cultured 2 days ex vivo,yes vs no,6/52,5/52,1/1/1,0,,152,,212,,0
```

## §4b Leading antigens across every stratum

**What:** each leading antigen's tumour-cell log2FC between the levels of every field.

**How:** pooled, donor-balanced, first level over second.

**Full table on disk:** `tables/tgt46_strat_joined.csv`

```csv
gene,B stage,ECP,HDACi,age (median 70),blood tumour % (median 59),burden (median 0.90),cultured 2 days ex vivo,disease,interferon,retinoid,sex,stage IV vs I-III,stage class,treatment-naive
CD70,0.581,-0.394,0.855,-0.0942,0.0992,-0.1,0.38,-0.36,-0.752,-0.691,0.0816,0.697,1.01,0.563
CPM,0.341,-0.0539,0.819,0.209,-0.572,-0.745,-1.03,-0.151,0.28,0.546,0.172,0.0381,0.525,-0.531
EPCAM,-1.24,-0.241,0.712,-0.0458,0.173,-0.855,0.304,-0.272,-0.562,-0.31,0.411,0.156,-1.19,0.598
FCRL3,-0.0844,0.514,0.475,0.232,-0.251,-0.389,-1.69,0.263,-0.531,0.432,0.0145,-0.683,-0.965,0.0485
FGFR1,0.0401,0.305,-0.105,-0.0596,-0.00348,-0.145,-0.54,0.0961,0.702,0.296,-0.258,-0.531,0.507,0.493
GPR15,0.339,0.239,1.05,-0.181,-0.123,-0.0327,-1.75,-0.111,-1.13,-0.125,-0.235,-0.00641,0.0945,-0.832
KIR3DL2,0.484,-0.216,0.448,-0.643,0.0339,0.243,-0.137,-0.15,-0.348,-0.673,0.0807,0.771,0.422,0.129
NRROS,0.574,-0.264,-0.166,-0.46,0.472,-0.0197,-0.00391,-0.313,-0.417,0.0365,0.0548,0.867,1.29,0.463
PI16,0.217,-0.553,-0.687,-0.286,-0.362,-0.0886,-0.609,-0.304,0.274,0.495,0.462,0.595,1.4,0.957
PTH2R,0.133,-0.282,0.871,-0.0534,-0.483,-0.0314,-0.527,-0.311,-0.259,0.0155,-0.132,0.593,0.618,-0.87
RAB25,-0.153,-0.112,0.987,-0.797,1.14,0.347,0.0118,-0.507,-0.539,-0.349,0.0764,0.21,-0.914,-0.199
STAM,0.214,0.253,-0.177,0.304,0.518,-0.00583,-0.564,-0.469,0.0652,0.231,0.403,-0.231,-0.378,0.39
TIGIT,-0.0237,0.29,0.313,-0.128,0.226,-0.154,0.246,0.214,0.0307,0.217,0.0018,-0.0789,-0.104,-0.615
TPO,0.225,0.368,0.326,0.0942,0.0393,0.0995,-2.68,-0.998,0.0726,0.713,0.542,0.706,1.06,-0.0991
TSPAN2,0.417,-0.199,0.31,-0.748,0.119,-0.297,-1.43,0.161,-0.202,-0.0845,-0.216,0.524,0.689,0.313
```

## §4e Sensitivity

**What:** whether the disease and ECP contrasts survive dropping the cultured and shared-clone donors.

**How:** the same contrast refitted; rho is against the full-cohort per-gene values.

**Full table on disk:** `tables/tgt46_sensitivity_excl.csv`

```csv
field,n_mal_x/y,rho_tumour_lfc,rho_did,tumour q<.05 all,tumour q<.05 excl,in both
disease,7/36,0.851,0.853,0,0,0
ECP,23/21,0.867,0.841,0,0,0
```

## §5 Joint model readings

**What:** for each donor-level term, how many genes are tumour-specific, compartment-wide, opposite or reactive-only.

**How:** OLS per gene on donor pseudobulk log2(mean + 0.05), HC3 errors, BH per term, run separately on the malignant and reactive arms.

**Full table on disk:** `tables/tgt46_joint_model.csv`

```csv
index,is_MF,ecp,is_F,age_z
tumour-specific,1,0,0,0
```

======================== COPY TO HERE ==========================