> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 48 — Skin, pooled and stratified

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**, pooled rather than paired. Notebook 42 compares malignant and
reactive CD4 cells WITHIN each donor; this one pools all malignant CD4 cells against all reactive
CD4 cells, which lets donors with only one arm contribute and makes clinical strata comparable as
cell populations.

Cohort: 109 independent donor keys (75 with both arms, 10 with one arm,
21 healthy-skin donors). Non-CD4 clones, diagnoses outside the MF/SS spectrum and repeat
biopsies of one patient were removed in 42 section 0b. Malignancy call `mal_cluster`.

Weighting: donor-balanced (every donor contributes equal weight to its arm) is the headline;
cell-weighted is printed beside it as `*_cells`. Pooled means are of log1p CP10K.

Effect sizes: `log2fc` (log2 ratio of pooled means, pseudocount 0.05), `d_det` (difference in the
fraction of cells detected), `cohen_d`, and `auroc` (probability a random malignant cell exceeds a
random comparator cell; 0.5 = no separation, and it is capped by detection for sparse genes).
CIs are a 1000-draw donor bootstrap. The headline p-value is van Elteren (`q_blocked`):
donor-level means ranked within study.

Stratified contrasts, per field: (a) malignant vs reactive within each stratum, (b) malignant
cells of one stratum vs the other, (c) the same for reactive cells, (d) difference-in-differences.
p-values come from permuting donor labels within study (50,000 permutations, `q_study`), with
a free permutation printed beside them (`q_free`). A field is tier 1 (tested), 2 (described) or 3
(counted) according to how many donors each side has in each arm.

Skin-specific: the lesion that was biopsied (patch / plaque / tumour / erythroderma) is a
stratification the blood cohort does not have, and it is confounded with stage and with the
malignant burden. Everything is transcript; skin carries no CITE-seq.

## §0b Cohort by arm pattern

**What:** how many independent donors have both CD4 arms, one, or are healthy.

**How:** an arm counts when it holds >= 20 cells.

**Full table on disk:** `tables/tgt48_cohort.csv`

```csv
pattern,HC,MF,MF/SS,SS,All
healthy,21,0,0,0,21
malignant only,0,2,0,1,3
no CD4 arm,1,1,0,1,3
paired,0,67,2,6,75
reactive only,0,5,0,2,7
All,22,75,2,10,109
```

## §1 Pool composition

**What:** who each pool is actually made of, under each weighting.

**How:** Kish effective donors (sum(w)^2/sum(w^2)); top_study_share_cells is the share of the cell-weighted pool held by its largest study.

**Full table on disk:** `tables/tgt48_pool_composition.csv`

```csv
arm,n_donors,n_cells,kish_n_eff_cells,kish_n_eff_donor,top_donor,top_donor_share_cells,top_study_share_cells,top_study
CD4_malignant,78,187748,33.4,78,B5__PT09,0.0555,0.476,li2024
CD4_reactive,82,147233,11.4,82,Li2024_atlas__CTCL7,0.261,0.622,li2024
CD4_HC,21,4724,12.4,21,D13__SC374_HTO10,0.166,0.256,chennareddy2025
```

## §2a Pooled shortlist

**What:** the genes that pass every pooled gate: clean in 42, log2FC > 0.5 with the bootstrap CI above zero, van Elteren q < 0.05, not a loss marker or lineage control.

**How:** donor-balanced pooled contrast, malignant vs reactive CD4.

**Full table on disk:** `tables/tgt48_pooled_all_genes.csv`

```csv
index,gene,cd_name,log2fc,log2fc_lo,log2fc_hi,pct_a,pct_b,d_det,auroc,auroc_lo,cohen_d,log2fc_cells,log2fc_paired,q_blocked,log2fc_vs_hc,auroc_vs_hc,on_42_shortlist
1720,PTH2R,,1.61,1.16,2.04,0.106,0.0113,0.0946,0.547,0.531,0.364,2.27,0.53,0.016,1.99,0.552,False
478,CPM,,1.49,1.21,1.75,0.177,0.0387,0.138,0.569,0.551,0.401,1.53,1.02,0.000677,2.23,0.584,True
1258,MRC2,,1.44,1.14,1.71,0.173,0.0386,0.134,0.567,0.549,0.386,1.95,0.892,2.61e-05,2.19,0.582,False
325,CD9,,1.34,0.929,1.71,0.289,0.0928,0.196,0.597,0.566,0.456,2.11,1.08,4.13e-05,1.17,0.59,True
313,CD70,CD70,1.31,1.09,1.56,0.234,0.068,0.166,0.582,0.568,0.4,1.38,1.1,1.15e-08,2.13,0.604,True
2150,THY1,,1.3,0.801,1.7,0.0766,0.0116,0.065,0.532,0.518,0.283,1.23,0.0513,0.00498,1.5,0.535,False
471,COL6A3,,1.3,1.03,1.59,0.184,0.0558,0.128,0.564,0.547,0.357,1.67,0.811,1.69e-05,2.22,0.584,True
1879,SGCE,,1.24,0.956,1.49,0.0854,0.00923,0.0762,0.538,0.527,0.316,1.68,0.259,0.0253,1.47,0.542,False
1334,NRN1,,1.22,0.904,1.52,0.085,0.0125,0.0725,0.536,0.526,0.29,1.54,0.776,3.89e-06,1.42,0.539,False
1663,PLD1,,1.2,0.943,1.43,0.113,0.0244,0.0887,0.544,0.532,0.299,1.32,0.736,0.000245,1.42,0.548,False
1361,OPCML,,1.14,0.411,1.75,0.0387,0.00269,0.036,0.518,0.507,0.233,1.5,0,0.0316,1.23,0.519,False
606,EPCAM,,1.03,0.529,1.49,0.0418,0.00289,0.0389,0.519,0.509,0.244,1.2,0.0128,0.0406,1.08,0.52,False
1950,SLC29A1,,1.02,0.878,1.16,0.115,0.0295,0.0859,0.543,0.534,0.275,1.01,0.841,6.99e-10,1.43,0.552,False
384,CHRNA6,,0.986,0.686,1.26,0.061,0.0108,0.0502,0.525,0.517,0.234,0.791,0.429,0.022,1.28,0.529,False
61,ADGRE1,,0.967,0.75,1.19,0.0795,0.0139,0.0656,0.533,0.524,0.253,0.927,0.792,0.00177,1.36,0.539,False
877,GRIK2,,0.909,0.437,1.33,0.0338,0.000999,0.0328,0.516,0.507,0.226,0.801,0,0.0133,0.948,0.517,False
1993,SLC4A2,,0.896,0.75,1.04,0.162,0.0571,0.105,0.551,0.541,0.261,1.01,0.698,1.31e-08,1.69,0.572,True
653,FAM174B,,0.896,0.634,1.13,0.0718,0.017,0.0548,0.527,0.519,0.22,1.22,0.491,0.0442,1.11,0.532,False
2229,TNFRSF8,CD30,0.889,0.52,1.22,0.0758,0.0239,0.0519,0.526,0.515,0.209,0.98,0.332,0.00544,1.63,0.537,False
699,FLT1,,0.876,0.559,1.16,0.0538,0.0114,0.0424,0.521,0.512,0.206,0.852,0.253,0.0257,1.13,0.525,False
1746,QSOX2,,0.846,0.621,1.1,0.0986,0.0312,0.0675,0.533,0.524,0.217,1.15,0.512,4.45e-06,1.46,0.545,False
907,HHLA2,,0.831,0.524,1.09,0.0408,0.00334,0.0375,0.519,0.511,0.219,0.759,0.0896,0.00316,0.911,0.52,False
2233,TNFSF4,,0.83,0.497,1.1,0.0722,0.0218,0.0504,0.525,0.515,0.199,0.631,0.289,0.0157,0.956,0.528,False
2138,TFRC,,0.821,0.655,1,0.341,0.162,0.179,0.585,0.57,0.327,1.13,0.731,3.19e-05,1.33,0.616,True
1976,SLC39A6,,0.818,0.632,0.997,0.223,0.0942,0.129,0.562,0.55,0.27,0.829,0.707,6.38e-10,1.4,0.586,True
2156,TLL1,,0.817,0.425,1.2,0.0341,0.00288,0.0312,0.516,0.509,0.196,1.41,0.214,8.4e-06,0.909,0.517,False
1928,SLC1A4,,0.812,0.51,1.06,0.198,0.0933,0.105,0.552,0.534,0.256,0.717,0.69,0.000997,1.9,0.585,True
888,GRN,,0.795,0.634,0.955,0.258,0.113,0.145,0.569,0.555,0.283,1.01,0.778,2.98e-07,1.64,0.602,True
1974,SLC39A14,,0.776,0.573,1.01,0.0844,0.0248,0.0596,0.53,0.521,0.203,0.896,0.566,1.86e-07,0.806,0.532,False
256,CD151,,0.771,0.587,0.943,0.237,0.103,0.134,0.564,0.55,0.266,1.12,0.794,1.15e-05,1.2,0.584,True
636,EVC2,,0.765,0.606,0.928,0.0787,0.0229,0.0558,0.528,0.521,0.197,1.16,0.624,1.25e-06,1.24,0.536,False
398,CLDN1,,0.759,0.445,1.06,0.0423,0.00876,0.0336,0.517,0.509,0.181,0.881,0.102,0.0205,0.218,0.509,False
1709,PTCHD1,,0.756,0.455,1.04,0.0296,0.0018,0.0278,0.514,0.508,0.203,0.851,0.0235,0.00626,0.826,0.515,False
748,GALNT2,,0.747,0.491,0.972,0.169,0.076,0.0928,0.545,0.532,0.222,1.03,0.388,0.000833,1.66,0.573,False
718,FUT8,,0.742,0.57,0.902,0.248,0.119,0.129,0.562,0.547,0.26,1.27,0.635,0.000504,1.84,0.605,True
1731,PTPRK,,0.722,0.458,0.984,0.0555,0.013,0.0425,0.521,0.513,0.188,0.75,0.185,0.000566,0.88,0.524,False
228,CALU,,0.677,0.572,0.783,0.227,0.105,0.122,0.558,0.548,0.231,0.532,0.587,2.96e-08,1.28,0.584,True
2004,SLC5A3,,0.671,0.52,0.828,0.254,0.131,0.122,0.558,0.546,0.237,0.686,0.617,0.00021,1.73,0.602,True
1214,MERTK,,0.659,0.436,0.908,0.03,0.00374,0.0263,0.513,0.509,0.185,0.711,0.246,0.000163,0.765,0.515,False
1151,LRFN4,,0.648,0.467,0.842,0.0697,0.0224,0.0473,0.523,0.517,0.165,0.939,0.351,2.85e-05,1.06,0.531,False
1299,NEO1,,0.645,0.449,0.836,0.0616,0.019,0.0426,0.521,0.515,0.165,0.65,0.492,0.000178,0.92,0.526,False
828,GPR158,,0.641,0.472,0.813,0.0274,0.00119,0.0262,0.513,0.509,0.195,0.838,0.187,0.000167,0.686,0.514,False
1275,MUC16,,0.635,0.313,0.976,0.0352,0.00848,0.0267,0.513,0.507,0.158,0.881,0.0275,0.0412,0.903,0.517,False
1922,SLC19A1,,0.63,0.463,0.798,0.082,0.029,0.053,0.526,0.519,0.165,0.737,0.393,0.00223,1.28,0.538,False
1983,SLC43A3,,0.629,0.447,0.818,0.13,0.0575,0.0725,0.535,0.525,0.18,0.767,0.299,0.00667,1.15,0.551,False
485,CRELD2,,0.629,0.406,0.861,0.25,0.135,0.115,0.555,0.538,0.225,0.839,0.307,6.11e-05,1.26,0.587,False
1117,LGALS3,,0.626,0.329,0.881,0.492,0.326,0.166,0.589,0.545,0.348,0.738,0.663,0.0327,0.608,0.589,True
2227,TNFRSF25,,0.622,0.399,0.86,0.353,0.189,0.164,0.575,0.552,0.264,0.894,0.698,0.0204,1.64,0.634,True
2315,VDAC1,,0.618,0.512,0.713,0.475,0.292,0.183,0.588,0.572,0.337,0.503,0.584,3.32e-06,1.34,0.652,True
1812,SCN2A,,0.615,0.285,0.933,0.024,0.000555,0.0235,0.512,0.504,0.19,0.596,0,0.0412,0.64,0.512,False
1346,NTRK1,,0.615,0.292,0.958,0.0459,0.0132,0.0326,0.516,0.508,0.153,0.743,0.15,0.0288,0.732,0.519,False
1674,PLXNB2,,0.613,0.288,0.911,0.0438,0.0156,0.0282,0.514,0.507,0.144,0.774,0.169,0.0123,1.14,0.521,False
2269,TSPAN3,,0.606,0.452,0.763,0.119,0.0483,0.0707,0.534,0.526,0.17,0.859,0.574,0.000149,1.31,0.551,False
2117,SYPL1,,0.602,0.467,0.727,0.377,0.208,0.17,0.577,0.561,0.272,0.508,0.599,0.000254,0.834,0.596,True
2268,TSPAN2,,0.599,0.353,0.833,0.155,0.0741,0.081,0.539,0.527,0.176,0.744,0.53,0.0288,1.02,0.554,True
2280,TTYH3,,0.599,0.436,0.747,0.0806,0.0313,0.0493,0.524,0.517,0.154,1.16,0.367,0.00359,1.34,0.537,False
542,DMXL2,,0.598,0.418,0.794,0.0458,0.0139,0.0319,0.516,0.511,0.152,0.908,0.391,1.55e-06,0.997,0.522,False
2106,STXBP1,,0.585,0.394,0.787,0.0588,0.0193,0.0394,0.52,0.514,0.147,0.668,0.345,0.00334,0.761,0.522,False
2104,STX6,,0.579,0.467,0.7,0.161,0.0756,0.0852,0.541,0.533,0.174,0.592,0.578,8.75e-06,1.3,0.564,True
114,AP2S1,,0.567,0.467,0.662,0.368,0.21,0.158,0.571,0.557,0.257,0.577,0.579,0.000399,1.5,0.632,True
1828,SDC4,,0.565,0.369,0.765,0.119,0.0525,0.0667,0.532,0.523,0.153,0.763,0.453,0.00164,0.976,0.543,False
1071,KIAA1324L,,0.564,0.334,0.779,0.059,0.0213,0.0377,0.519,0.512,0.141,0.841,0.201,0.0172,0.789,0.523,False
54,ADCY3,,0.563,0.322,0.789,0.111,0.0542,0.0568,0.528,0.518,0.149,1.09,0.204,0.0315,1.63,0.55,False
1637,PDIA4,,0.561,0.432,0.692,0.29,0.157,0.133,0.561,0.549,0.218,0.0565,0.637,0.00172,1.26,0.599,True
1570,P2RY11,,0.558,0.345,0.795,0.0874,0.0378,0.0496,0.524,0.517,0.14,1.2,0.405,0.00118,1.27,0.538,False
1003,IL6R,,0.557,0.376,0.744,0.234,0.121,0.113,0.552,0.539,0.193,0.547,0.614,0.0234,1.06,0.576,True
1229,MGAT4B,,0.554,0.383,0.739,0.144,0.0712,0.0733,0.535,0.525,0.162,0.794,0.409,0.00019,1.54,0.562,False
348,CDON,,0.552,0.3,0.848,0.0268,0.00303,0.0238,0.512,0.507,0.163,0.676,0.0716,0.0148,0.578,0.512,False
855,GPR55,,0.551,0.267,0.821,0.0915,0.0399,0.0516,0.525,0.515,0.142,0.614,0.389,0.0397,0.801,0.532,False
1097,L1CAM,,0.55,0.253,0.869,0.0648,0.0237,0.041,0.52,0.512,0.138,0.63,0.293,0.0345,0.5,0.52,False
30,ACVR2A,,0.547,0.292,0.789,0.0883,0.0417,0.0466,0.523,0.513,0.14,1.03,0.338,0.0237,0.869,0.531,False
1745,QSOX1,,0.544,0.403,0.682,0.169,0.0819,0.0872,0.541,0.532,0.17,0.707,0.584,6.91e-06,1.39,0.569,True
401,CLDN15,,0.543,0.386,0.688,0.0514,0.0193,0.0321,0.516,0.511,0.132,0.878,0.345,0.000115,1.12,0.524,False
232,CASD1,,0.539,0.356,0.732,0.115,0.0544,0.0606,0.53,0.522,0.149,1.02,0.414,0.00448,1.43,0.551,False
1877,SGCB,,0.539,0.408,0.649,0.0957,0.0384,0.0572,0.528,0.522,0.145,0.527,0.615,1.94e-05,0.988,0.538,False
1244,MMP17,,0.536,0.209,0.883,0.0199,0.00273,0.0172,0.509,0.504,0.145,0.577,0.0381,0.0028,0.561,0.509,False
968,IGSF3,,0.532,0.333,0.752,0.0279,0.00458,0.0233,0.512,0.507,0.153,0.563,0.213,6.77e-06,0.629,0.513,False
540,DLL3,,0.528,0.359,0.707,0.0308,0.00414,0.0267,0.513,0.509,0.173,0.643,0.227,0.000313,0.495,0.513,False
1022,ITGA3,,0.528,0.244,0.784,0.0573,0.0247,0.0325,0.516,0.509,0.127,0.712,0.234,0.0171,1.14,0.526,False
2059,SORCS1,,0.525,0.35,0.701,0.0224,0.00256,0.0198,0.51,0.506,0.15,0.688,0.0281,0.00152,0.503,0.51,False
1671,PLXNA3,,0.52,0.33,0.743,0.091,0.0421,0.049,0.524,0.517,0.133,0.822,0.377,0.00149,0.66,0.529,False
930,HRH1,,0.515,0.277,0.762,0.0303,0.00675,0.0236,0.512,0.506,0.148,0.531,0.138,0.00669,0.686,0.514,False
719,FZD1,,0.515,0.361,0.665,0.0231,0.0018,0.0213,0.511,0.507,0.163,0.561,0.103,0.00104,0.584,0.511,False
2016,SLC7A1,,0.509,0.38,0.638,0.146,0.0753,0.0707,0.534,0.526,0.152,0.573,0.467,0.00021,1.36,0.56,False
319,CD82,,0.505,0.349,0.655,0.471,0.307,0.164,0.575,0.554,0.274,0.56,0.632,0.000419,1.17,0.639,True
```

## §2a Controls

**What:** the antigens with a known answer, pooled.

**How:** same contrast. CCR4, TNFRSF8, KIR3DL2 and IL2RA should be up; CD7 and DPP4 down with AUROC below 0.5.

```csv
gene,log2fc,log2fc_lo,log2fc_hi,d_det,auroc,auroc_lo,auroc_hi,log2fc_paired,q_blocked
CCR4,0.562,0.218,0.887,0.115,0.554,0.529,0.58,0.324,0.973
TNFRSF8,0.889,0.52,1.22,0.0519,0.526,0.515,0.538,0.332,0.00544
KIR3DL2,1.63,1.16,2.02,0.123,0.561,0.538,0.586,0,0.775
IL2RA,-0.0975,-0.364,0.159,0.0182,0.505,0.491,0.521,-0.172,0.346
CD7,-1.73,-2.29,-1.28,-0.353,0.316,0.287,0.349,-2.85,1.19e-11
DPP4,-0.342,-0.797,0.035,-0.016,0.491,0.477,0.506,-0.841,0.000346
CD5,-0.0115,-0.261,0.234,0.0411,0.51,0.484,0.538,-0.0584,0.753
CD6,0.149,-0.0436,0.343,0.082,0.529,0.504,0.557,0.0957,0.463
CD27,0.129,-0.165,0.401,0.0421,0.518,0.488,0.549,-0.359,0.359
CD2,-0.252,-0.428,-0.084,-0.0392,0.449,0.417,0.48,-0.146,0.24
```

## §2b Pooled vs paired agreement

**What:** whether pooling reproduces 42's paired ranking, and where it does not.

**How:** Spearman over all genes, then the largest absolute disagreements.

```csv
key,value
spearman_pooled_vs_paired,0.739
spearman_donor_balanced_vs_cell_weighted,0.791
```

## §2c Cell-level separation and leave-one-study-out

**What:** how well each leading antigen separates single cells, and whether any one study carries the effect.

**How:** pooled AUROC with a donor-bootstrap CI; then the pooled log2FC recomputed with each study dropped in turn.

**Full table on disk:** `tables/tgt48_loso.csv`

```csv
gene,pooled,loso_min,loso_max,study_moving_most,sign_kept_every_study
PTH2R,1.61,1.46,1.74,brentuximab2026,True
CPM,1.49,1.4,1.54,brentuximab2026,True
MRC2,1.44,1.29,1.48,li2024,True
CD9,1.34,0.909,1.44,li2024,True
CD70,1.31,1.23,1.36,alkon2024,True
THY1,1.3,1.09,1.37,li2024,True
COL6A3,1.3,0.936,1.42,li2024,True
SGCE,1.24,1.07,1.33,brentuximab2026,True
NRN1,1.22,1.08,1.27,li2024,True
PLD1,1.2,1.1,1.25,li2024,True
OPCML,1.14,0.56,1.49,il4ra2026,True
EPCAM,1.03,0.905,1.1,li2024,True
SLC29A1,1.02,0.978,1.04,gaydosik2022,True
CHRNA6,0.986,0.91,1.19,li2024,True
ADGRE1,0.967,0.904,1.02,chennareddy2025,True
CCR4,0.562,0.475,0.802,li2024,True
TNFRSF8,0.889,0.5,0.946,li2024,True
KIR3DL2,1.63,1.35,1.94,li2024,True
IL2RA,-0.0975,-0.134,-0.0367,brentuximab2026,True
CD7,-1.73,-1.83,-1.64,gaydosik2019,True
DPP4,-0.342,-0.475,-0.277,li2024,True
```

## §3 Power table

**What:** donors per arm on each side of every candidate split, and the resulting tier.

**How:** tier 1 = tested, 2 = described, 3 = counted only.

**Full table on disk:** `tables/tgt48_power_table.csv`

```csv
field,x,y,n_mal_x,n_mal_y,tier_mal,n_rea_x,n_rea_y,tier_rea,tier_did
lesion,tumour/eryth,patch/plaque,31,19,1,34,22,1,1
disease,MF,SS,69,7,1,72,8,1,1
stage class,advanced,early,56,20,1,59,20,1,1
stage IV vs I-III,IV,I-III,20,56,1,22,57,1,1
B stage,B2,B0-B1,9,45,1,10,49,1,1
CD30,pos,neg,9,8,1,9,8,1,1
large-cell transformation,yes,no,23,9,1,23,9,1,1
folliculotropic,yes,no,5,0,3,5,0,3,3
any systemic tx at sampling,yes,no,15,37,1,16,42,1,1
retinoid,yes,no,4,48,2,5,53,1,2
ECP,yes,no,7,45,1,7,51,1,1
brentuximab,yes,no,1,51,3,1,57,3,3
treatment-naive,yes,no,6,47,1,7,49,1,1
sex,F,M,32,46,1,32,50,1,1
age (median 65),older,younger,39,39,1,42,40,1,1
burden (median 0.65),high,low,43,35,1,40,42,1,1
cultured ex vivo,yes,no,4,74,2,5,77,1,2
```

## §4a Stratified summary

**What:** how many genes differ between the two levels of each field, for tumour cells, reactive cells and the difference-in-differences.

**How:** within-study donor permutation (q_study) and free permutation (q_free), BH.

**Full table on disk:** `tables/tgt48_strat_summary.csv`

```csv
field,x vs y,mal n,rea n,tier mal/rea/did,log10 cfg study,tumour q_study,tumour q_free,reactive q_study,reactive q_free,DiD q_study,DiD q_free
lesion,tumour/eryth vs patch/plaque,31/19,34/22,1/1/1,4.28,0,1,0,0,0,0
disease,MF vs SS,69/7,72/8,1/1/1,1.73,0,0,0,0,0,34
stage class,advanced vs early,56/20,59/20,1/1/1,8.67,0,3,0,0,0,2
stage IV vs I-III,IV vs I-III,20/56,22/57,1/1/1,8.73,0,0,1,0,0,0
B stage,B2 vs B0-B1,9/45,10/49,1/1/1,4.18,0,1,0,0,0,0
CD30,pos vs neg,9/8,9/8,1/1/1,1.38,0,0,0,0,0,0
large-cell transformation,yes vs no,23/9,23/9,1/1/1,0.903,,166,,285,,0
folliculotropic,yes vs no,5/0,5/0,3/3/3,,,,,,,
any systemic tx at sampling,yes vs no,15/37,16/42,1/1/1,6.08,0,0,0,0,0,0
retinoid,yes vs no,4/48,5/53,2/1/2,,,,0,0,,
ECP,yes vs no,7/45,7/51,1/1/1,2.38,0,0,0,0,0,0
brentuximab,yes vs no,1/51,1/57,3/3/3,,,,,,,
treatment-naive,yes vs no,6/47,7/49,1/1/1,3.38,0,0,0,0,0,0
sex,F vs M,32/46,32/50,1/1/1,14.4,0,0,0,0,0,0
age (median 65),older vs younger,39/39,42/40,1/1/1,13.5,0,0,0,0,0,0
burden (median 0.65),high vs low,43/35,40/42,1/1/1,15.9,0,0,0,0,2,0
cultured ex vivo,yes vs no,4/74,5/77,2/1/2,,,,0,83,,
```

## §4b Leading antigens across every stratum

**What:** each leading antigen's tumour-cell log2FC between the levels of every field.

**How:** pooled, donor-balanced, first level over second.

**Full table on disk:** `tables/tgt48_strat_joined.csv`

```csv
gene,B stage,CD30,ECP,age (median 65),any systemic tx at sampling,brentuximab,burden (median 0.65),cultured ex vivo,disease,large-cell transformation,lesion,retinoid,sex,stage IV vs I-III,stage class,treatment-naive
ADGRE1,0.229,0.169,0.264,0.0208,-0.109,0.0921,-0.12,0.333,-0.303,0.286,0.209,0.34,0.282,0.154,0.397,0.156
CD70,-0.222,0.726,0.262,-0.192,-0.254,-1.97,-0.313,0.00583,0.0925,-0.21,0.0217,-0.246,-0.0214,-0.12,-0.143,0.313
CD9,-0.192,0.701,0.139,0.0854,-0.174,-1.4,-0.269,0.0384,1.11,-0.0677,-0.487,0.497,-0.425,-0.772,-0.521,-0.207
CHRNA6,-0.124,-0.361,0.0686,-0.543,0.192,-1.08,-0.309,-0.848,0.114,-0.157,-0.655,-0.47,0.769,0.449,0.129,-0.015
COL6A3,-0.227,1.27,-0.603,-1.27,-0.126,1.74,0.527,-1.83,0.721,1.02,-0.709,-0.172,0.787,0.249,-0.607,-0.38
CPM,-0.339,0.742,0.0646,-0.281,0.389,1.65,0.115,-0.205,-0.267,0.788,0.0859,0.329,0.71,0.0803,0.315,-0.719
EPCAM,-0.0424,-0.153,-1.12,-0.714,0.261,2.38,0.715,-1.19,1.13,1.65,0.601,-1.03,-0.00335,0.492,0.932,-0.29
MRC2,-0.889,1.09,-0.943,-0.692,-0.173,1.31,0.655,-0.356,1.55,1.56,-0.115,-0.298,0.511,-0.151,-0.167,-0.859
NRN1,0.106,0.714,0.143,-0.383,0.183,1.39,1.08,-1.16,0.211,0.942,0.385,0.499,-0.246,0.141,0.19,-0.584
OPCML,-1.39,0.236,0.611,1.22,-0.247,-1.42,0.4,3.17,-0.836,0.0917,1.28,-1.29,1.38,-1.1,0.644,-0.118
PLD1,-1.36,1.47,-0.68,-0.723,0.00119,1.76,0.693,0.4,1.33,0.653,-0.0791,-0.54,0.487,-0.716,-0.668,0.0894
PTH2R,0.0796,1.07,1.13,-0.0921,0.324,-1.85,0.549,-1.14,-1.45,0.986,0.415,-0.88,0.781,0.471,1.17,-0.424
SGCE,-0.277,1.22,-0.471,-0.787,0.00234,1.68,0.448,-1.2,-0.183,1.33,0.517,-1.02,0.614,-0.0247,0.762,0.102
SLC29A1,-0.612,-0.477,-0.586,0.0601,-0.419,-0.355,0.159,0.183,0.593,0.105,0.0866,-0.311,-0.0846,0.134,0.00956,0.329
THY1,-1.49,-0.204,-1.17,0.212,-0.971,0.165,-0.0587,0.834,1.54,0.0614,-1.29,-0.438,0.407,-1.17,-0.839,-0.068
```

## §4e Sensitivity

**What:** whether the lesion and disease contrasts survive dropping the cultured donors and dropping li2024.

**How:** the same contrast refitted on the reduced cohort; rho is against the full-cohort per-gene values.

**Full table on disk:** `tables/tgt48_sensitivity_excl.csv`

```csv
field,excluded,n_excluded,n_mal_x/y,rho_tumour_lfc,rho_did,tumour q<.05 all,tumour q<.05 excl,in both
lesion,cultured ex vivo,5,28/19,0.924,0.912,0,0,0
lesion,li2024,35,26/17,0.864,0.773,0,0,0
```

## §5 Joint model readings

**What:** for each donor-level term, how many genes are tumour-specific, compartment-wide, opposite or reactive-only.

**How:** OLS per gene on donor pseudobulk log2(mean + 0.05), HC3 errors, BH per term, run separately on the malignant and reactive arms.

**Full table on disk:** `tables/tgt48_joint_model.csv`

```csv
index,is_tumour_lesion,is_MF,is_advanced,is_F,age_z
compartment-wide,0,11,0,0,0
opposite,0,1,0,0,0
reactive-only,0,132,0,0,0
tumour-specific,0,73,3,0,1
```

## §5 Leading antigens, malignant-arm coefficients

**What:** the size of each donor-level effect on the leading antigens.

**How:** log2 units per level, or per SD of age.

```csv
gene,b_is_tumour_lesion_mal,b_is_MF_mal,b_is_advanced_mal,b_is_F_mal,b_age_z_mal,q_is_tumour_lesion_mal,q_is_MF_mal,q_is_advanced_mal,q_is_F_mal,q_age_z_mal
PTH2R,0.0236,-0.3,0.964,0.233,-0.156,0.991,0.805,0.71,0.976,0.871
CPM,0.25,-0.353,0.237,0.645,-0.126,0.9,0.738,0.892,0.976,0.896
MRC2,-0.0392,0.979,0.406,0.373,-0.439,0.991,0.0339,0.838,0.976,0.607
CD9,-0.347,0.721,-0.0115,-0.238,0.13,0.871,0.573,0.996,0.976,0.912
CD70,0.405,0.264,-0.172,-0.053,0.0198,0.86,0.81,0.911,0.984,0.981
THY1,-0.709,0.425,0.269,0.379,0.0455,0.86,0.594,0.884,0.976,0.955
COL6A3,-0.458,-0.46,0.276,0.346,-0.37,0.86,0.26,0.892,0.976,0.784
SGCE,0.345,-0.00116,0.516,0.491,-0.247,0.863,1,0.813,0.976,0.83
NRN1,-0.0807,0.0689,0.453,0.0569,-0.00968,0.972,0.961,0.838,0.984,0.992
PLD1,-0.262,0.906,-0.266,0.217,-0.0493,0.863,0.0122,0.869,0.976,0.958
OPCML,-0.177,0.434,0.207,0.346,0.115,0.911,0.21,0.869,0.976,0.87
EPCAM,0.294,0.609,0.292,0.0453,-0.0333,0.863,0.164,0.838,0.984,0.965
SLC29A1,-0.181,0.297,0.179,0.0255,0.0967,0.863,0.602,0.873,0.984,0.871
CHRNA6,-0.165,-0.246,0.438,0.515,-0.203,0.9,0.646,0.826,0.976,0.87
ADGRE1,0.205,-0.571,0.305,0.252,-0.00258,0.874,0.153,0.838,0.976,0.997
```

======================== COPY TO HERE ==========================