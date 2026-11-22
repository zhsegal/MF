> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 45 — Blood CITE-seq: measured protein and what explains it

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **blood**, CITE-seq. This is the only part of the project that checks the
transcript ranking against MEASURED surface protein on the same cells:
61,215 buus2025 cells carry a 124-marker antibody-derived-tag panel
plus 4 isotype controls, over 17 donors (15 Sezary and 2 MF by
the curated clin_dx -- not all Sezary). No healthy donor and no skin sample carries ADT.

Marker gate: a marker counts as measured only if its mean log1p count exceeds that donor's mean
isotype count by >= 0.10 in at least two thirds of donors, and it is detected in >= 1% of cells.
25 of 124 markers pass. This matters: ADT_CD158k (the KIR3DL2 clone, the
lacutamab target and the top transcript hit of the blood screen) sits at isotype level, so its
near-zero protein delta is a clone that did not bind, NOT protein contradicting transcript.

Normalisation: CLR (log1p count minus the cell's mean log1p over the non-control markers). Two
alternatives were compared on markers whose direction is known and all three agree, so the
choice is not carrying the result.

Protein delta: malignant minus reactive CD4, per donor, median over the 17 donors
with both arms. The transcript side is recomputed on the SAME donors so the two columns describe
the same patients.

Trafficking model: per marker, inside malignant CD4 cells, protein_CLR ~ own mRNA + 11
trafficking module scores + RNA depth + total ADT depth, donor-demeaned (and as a mixed model
with donor random intercept for the focus markers). The reported quantity is an INCREMENTAL R2:
what the machinery explains after the marker's own mRNA and library size have had their turn.
§4e tests it out of sample -- fit on 16 donors, score the 17th -- against 200 draws of
size-matched random gene programs, because 11 extra covariates on 46k cells always explain
something in sample.

Caveats that apply throughout: the modules are averaged over HVG survivors only and three rest on
fewer than five genes; they are strongly correlated with each other and with how much of anything
a cell is making, so read the trafficking column as "the cell's machinery", not as a named
pathway.

## §1 Marker gate

**What:** which panel markers were measured above isotype background, and which were not.

**How:** per-donor mean log1p against that donor's mean isotype log1p; `passes` is the gate.

**Full table on disk:** `tables/tgt45_adt_gate.csv`

```csv
index,frac_cells_pos,mean_log1p,mean_log1p_minus_bg,frac_donors_above_bg,is_control,gene,passes
ADT_CD1a,0.0125,0.0163,-0.0149,0,False,CD1A,False
ADT_CD1c,0.00959,0.0162,-0.0149,0,False,CD1C,False
ADT_CD1d,0.0498,0.0735,0.0423,0.0588,False,CD1D,False
ADT_CD2,0.741,1.8,1.77,0.941,False,CD2,True
ADT_CD3,0.679,1.34,1.3,0.941,False,CD3E,True
ADT_CD4,0.84,2.22,2.18,1,False,CD4,True
ADT_CD5,0.673,1.11,1.08,1,False,CD5,True
ADT_CD7,0.488,1.03,1,0.941,False,CD7,True
ADT_CD8A,0.125,0.321,0.29,0.529,False,CD8A,False
ADT_CD11a,0.459,0.705,0.674,0.765,False,ITGAL,True
ADT_CD11a_CD18,0.0282,0.0484,0.0172,0,False,,False
ADT_CD11b,0.0288,0.0402,0.00904,0,False,ITGAM,False
ADT_CD11c,0.02,0.0277,-0.00347,0.0588,False,ITGAX,False
ADT_CD14,0.00879,0.0106,-0.0206,0,False,CD14,False
ADT_CD16,0.0769,0.14,0.109,0.0588,False,FCGR3A,False
ADT_CD18,0.34,0.44,0.409,0.647,False,ITGB2,False
ADT_CD19,0.0333,0.0442,0.0131,0,False,CD19,False
ADT_CD20,0.0308,0.0402,0.009,0,False,MS4A1,False
ADT_CD21,0.0329,0.0463,0.0151,0,False,CR2,False
ADT_CD22,0.0442,0.0639,0.0328,0.0588,False,CD22,False
ADT_CD23,0.00965,0.0151,-0.0161,0,False,FCER2,False
ADT_CD25,0.0607,0.0688,0.0376,0.118,False,IL2RA,False
ADT_CD26,0.23,0.331,0.3,0.824,False,DPP4,True
ADT_CD27,0.8,2.18,2.15,1,False,CD27,True
ADT_CD28,0.583,1.39,1.36,1,False,CD28,True
ADT_CD29,0.297,0.379,0.348,0.765,False,ITGB1,True
ADT_CD30,0.047,0.076,0.0448,0.176,False,TNFRSF8,False
ADT_CD31,0.0738,0.0846,0.0534,0.0588,False,PECAM1,False
ADT_CD35,0.00475,0.00485,-0.0263,0,False,CR1,False
ADT_CD38,0.134,0.184,0.153,0.471,False,CD38,False
ADT_CD39,0.118,0.148,0.116,0.118,False,ENTPD1,False
ADT_CD43,0.0467,0.147,0.116,0.0588,False,SPN,False
ADT_CD44,0.433,0.67,0.639,0.706,False,CD44,True
ADT_CD45RA,0.376,0.629,0.598,0.882,False,PTPRC,True
ADT_CD45RO,0.338,0.636,0.605,0.941,False,PTPRC,True
ADT_CD48,0.641,1.39,1.35,0.706,False,CD48,True
ADT_CD49a,0.114,0.125,0.0936,0.235,False,ITGA1,False
ADT_CD49d,0.181,0.265,0.234,0.353,False,ITGA4,False
ADT_CD52,0.257,0.491,0.459,0.765,False,CD52,True
ADT_CD55,0.0602,0.0852,0.054,0,False,CD55,False
ADT_CD56,0.0162,0.0275,-0.00373,0.0588,False,NCAM1,False
ADT_CD56_QA17A16,0.0956,0.227,0.196,0.118,False,NCAM1,False
ADT_CD57,0.115,0.194,0.163,0.353,False,B3GAT1,False
ADT_CD62L,0.68,2.17,2.14,1,False,SELL,True
ADT_CD69,0.0846,0.102,0.0705,0.235,False,CD69,False
ADT_CD70,0.0956,0.116,0.0847,0.176,False,CD70,False
ADT_CD71,0.0454,0.0829,0.0517,0.176,False,TFRC,False
ADT_CD73,0.185,0.287,0.255,0.471,False,NT5E,False
ADT_CD80,0.0538,0.0701,0.0389,0.0588,False,CD80,False
ADT_CD84,0.0147,0.0204,-0.0107,0,False,CD84,False
ADT_CD86,0.0148,0.0203,-0.0109,0,False,CD86,False
ADT_CD95,0.195,0.227,0.195,0.353,False,FAS,False
ADT_CD99,0.0311,0.0359,0.00471,0.118,False,CD99,False
ADT_CD101,0.0372,0.0375,0.0063,0,False,CD101,False
ADT_CD103,0.0455,0.0666,0.0354,0,False,ITGAE,False
ADT_CD107a,0.036,0.0454,0.0142,0.0588,False,LAMP1,False
ADT_CD117,0.0145,0.0173,-0.0139,0,False,KIT,False
ADT_CD120b,0.242,0.372,0.34,0.647,False,TNFRSF1B,False
ADT_CD122,0.0922,0.121,0.0896,0.294,False,IL2RB,False
ADT_CD124,0.195,0.297,0.265,0.824,False,IL4R,True
ADT_CD126,0.0529,0.0708,0.0396,0.0588,False,IL6R,False
ADT_CD127,0.417,0.657,0.626,1,False,IL7R,True
ADT_CD134,0.136,0.192,0.161,0.647,False,TNFRSF4,False
ADT_CD137,0.114,0.151,0.12,0.588,False,TNFRSF9,False
ADT_CD137L,0.00854,0.0124,-0.0188,0,False,TNFSF9,False
ADT_CD138,0.0312,0.0451,0.014,0.0588,False,SDC1,False
ADT_CD141,0.0539,0.074,0.0428,0.118,False,THBD,False
ADT_CD150,0.0718,0.104,0.0732,0.294,False,SLAMF1,False
ADT_CD152,0.0255,0.0335,0.0023,0.118,False,CTLA4,False
ADT_CD154,0.105,0.168,0.137,0.647,False,CD40LG,False
ADT_CD158,0.0754,0.127,0.0957,0.471,False,,False
ADT_CD158b,0.0917,0.182,0.15,0.824,False,KIR2DL3,True
ADT_CD158e1,0.0432,0.0779,0.0468,0.118,False,KIR3DL1,False
ADT_CD158f,0.0721,0.128,0.097,0.471,False,KIR2DL5A,False
ADT_CD161,0.0959,0.139,0.107,0.294,False,KLRB1,False
ADT_CD163,0.0306,0.0445,0.0133,0.0588,False,CD163,False
ADT_CD164,0.0749,0.102,0.0708,0.176,False,CD164,False
ADT_CD183,0.0434,0.0852,0.054,0.118,False,CXCR3,False
ADT_CD184,0.0429,0.0704,0.0392,0.176,False,CXCR4,False
ADT_CD185,0.0294,0.0519,0.0207,0.0588,False,CXCR5,False
ADT_CD193,0.0123,0.0177,-0.0135,0.0588,False,CCR3,False
ADT_CD194,0.517,0.978,0.947,0.882,False,CCR4,True
ADT_CD195,0.0826,0.123,0.0921,0.176,False,CCR5,False
ADT_CD196,0.0981,0.12,0.0888,0.353,False,CCR6,False
ADT_CD197,0.126,0.165,0.134,0.765,False,CCR7,True
ADT_CD207,0.0194,0.0253,-0.00587,0.0588,False,CD207,False
ADT_CD223,0.0782,0.0942,0.063,0.118,False,LAG3,False
ADT_CD224,0.125,0.145,0.114,0.176,False,GGT1,False
ADT_CD244,0.138,0.261,0.23,0.235,False,CD244,False
ADT_CD254,0.0376,0.0486,0.0174,0.0588,False,TNFSF11,False
ADT_CD272,0.0391,0.055,0.0238,0.0588,False,BTLA,False
ADT_CD273,0.128,0.154,0.122,0.235,False,PDCD1LG2,False
ADT_CD274,0.0178,0.0214,-0.00975,0,False,CD274,False
ADT_CD275,0.1,0.14,0.109,0.176,False,ICOSLG,False
ADT_CD278,0.349,0.429,0.398,1,False,ICOS,True
ADT_CD279,0.375,0.539,0.508,0.941,False,PDCD1,True
ADT_CD294,0.0196,0.0255,-0.00565,0.0588,False,PTGDR2,False
ADT_CD307c,0.0979,0.186,0.154,0.118,False,FCRL3,False
ADT_CD314,0.106,0.175,0.144,0.118,False,KLRK1,False
ADT_CD319,0.0526,0.0598,0.0286,0.118,False,SLAMF7,False
ADT_CD326,0.0736,0.0873,0.0561,0.118,False,EPCAM,False
ADT_CD357,0.0596,0.0789,0.0477,0.0588,False,TNFRSF18,False
ADT_CD360,0.0107,0.0257,-0.00545,0,False,IL21R,False
ADT_CD366,0.08,0.104,0.0732,0.176,False,HAVCR2,False
ADT_APC,0.000114,9.99e-05,-0.0311,0,True,,False
ADT_Biotin,0.0256,0.0534,0.0222,0.0588,True,,False
ADT_CCR10,0.0424,0.0784,0.0472,0.0588,False,CCR10,False
ADT_CX3CR1,0.149,0.219,0.188,0.235,False,CX3CR1,False
ADT_FceRIa,0.0358,0.0546,0.0235,0.118,False,FCER1A,False
ADT_FITC,0.00498,0.00804,-0.0231,0,True,,False
ADT_GARP,0.121,0.146,0.115,0.529,False,LRRC32,False
ADT_GP130,0.0607,0.0899,0.0587,0.176,False,IL6ST,False
ADT_GPR56,0.0937,0.218,0.187,0.118,False,ADGRG1,False
ADT_HLA-DR,0.0896,0.127,0.0961,0.588,False,HLA-DRA,False
ADT_ITGB7,0.148,0.227,0.195,0.882,False,ITGB7,True
ADT_KLRG1,0.0279,0.043,0.0119,0.118,False,KLRG1,False
ADT_LAP,0.0257,0.0493,0.0181,0,False,TGFB1,False
ADT_PE,0.0468,0.0744,0.0432,0,True,,False
ADT_TCRab,0.191,0.336,0.305,0.353,False,,False
ADT_TCRgd,0.0172,0.024,-0.00715,0.0588,False,,False
ADT_TIGIT,0.155,0.205,0.174,0.706,False,TIGIT,True
ADT_TSLPR,0.0538,0.074,0.0428,0.0588,False,CRLF2,False
ADT_CD213a1,0.00595,0.00548,-0.0257,0,False,IL13RA1,False
ADT_CD213a2,0.00271,0.00274,-0.0284,0,False,IL13RA2,False
ADT_CD217,0.0025,0.00232,-0.0289,0,False,IL17RA,False
ADT_CLA,0.00727,0.00826,-0.0229,0,False,SELPLG,False
ADT_CD158k,0.000147,0.000282,-0.0309,0,False,KIR3DL2,False
ADT_CD8a,0.00689,0.0883,0.0571,0,False,CD8A,False
```

## §2 Normalisation check

**What:** the three candidate normalisations judged on markers whose direction is known (CD7, DPP4, CD2, CD5 down; CCR4 up).

**How:** malignant-minus-reactive delta per normalisation; correct_sign is the test.

**Full table on disk:** `tables/tgt45_normalisation_check.csv`

```csv
norm,gene,expected,delta,n_donors,p,correct_sign
clr,CD2,down,-0.73,17,0.00385,True
clr,CD5,down,0.229,17,0.132,False
clr,CD7,down,-1.03,17,0.000656,True
clr,DPP4,down,-0.612,17,3.05e-05,True
clr,CCR4,up,0.823,17,0.00934,True
bg,CD2,down,-0.749,17,0.00464,True
bg,CD5,down,0.191,17,0.159,False
bg,CD7,down,-1.03,17,0.000504,True
bg,DPP4,down,-0.62,17,4.58e-05,True
bg,CCR4,up,0.828,17,0.0079,True
old,CD2,down,-0.702,17,0.00385,True
old,CD5,down,0.197,17,0.12,False
old,CD7,down,-0.937,17,0.000504,True
old,DPP4,down,-0.554,17,3.05e-05,True
old,CCR4,up,0.731,17,0.00665,True
```

## §3 Protein against transcript, every mapped marker

**What:** the measured protein delta beside the transcript log2FC for the same donors.

**How:** paired Wilcoxon on the protein side (q_prot); `detected` marks the markers above isotype background -- the ones at background carry no direction and must be excluded from any agreement rate.

**Full table on disk:** `tables/tgt45_protein_vs_transcript.csv`

```csv
adt,gene,prot_delta,n_donors_prot,p_prot,detected,q_prot,log2fc_paired,frac_donors_up,q,cd_name,stage,surface_caveat
ADT_CD1a,CD1A,0.0261,17,0.132,False,0.272,0,0.0667,0.776,,,
ADT_CD1c,CD1C,0.0267,17,0.0984,False,0.244,0,0,1,,,
ADT_CD1d,CD1D,0.0336,17,0.0638,False,0.2,0,0,1,,,
ADT_CD2,CD2,-0.73,17,0.00385,True,0.0282,-0.802,0.2,0.022,,,
ADT_CD3,CD3E,-0.402,17,0.000381,True,0.00504,-0.173,0.267,0.175,,,
ADT_CD5,CD5,0.229,17,0.132,True,0.272,-0.0778,0.333,0.643,CD5,phase 1/2,
ADT_CD7,CD7,-1.03,17,0.000656,True,0.0071,-1.78,0.133,0.0251,CD7,phase 1/2,
ADT_CD8A,CD8A,-0.432,16,6.1e-05,False,0.00121,-2,0,0.022,,,
ADT_CD11a,ITGAL,-0.0244,14,1,True,1,0.123,0.533,0.424,,,
ADT_CD11b,ITGAM,0.0211,17,0.145,False,0.284,0.00485,0.533,0.633,,,
ADT_CD11c,ITGAX,0.0173,17,0.109,False,0.244,0,0.0667,0.731,,,
ADT_CD14,CD14,0.0239,17,0.159,False,0.292,0,0.133,0.325,,,
ADT_CD16,FCGR3A,0.00929,17,0.579,False,0.715,-0.199,0.0667,0.155,,,
ADT_CD18,ITGB2,-0.0177,14,0.903,False,0.927,0.581,0.8,0.0574,,,
ADT_CD19,CD19,0.00277,17,0.854,False,0.925,0,0.0667,0.776,,,
ADT_CD20,MS4A1,0.0138,17,0.284,False,0.412,0,0.4,0.952,,,
ADT_CD21,CR2,0.00799,17,0.747,False,0.846,0,0.4,0.636,,,
ADT_CD22,CD22,-0.00888,17,0.517,False,0.662,0,0.0667,0.456,,,
ADT_CD23,FCER2,0.0196,17,0.19,False,0.31,0,0.0667,0.456,,,
ADT_CD25,IL2RA,0.0221,17,0.0569,False,0.192,0.0111,0.533,0.584,CD25,approved (2024-08),"shed as sIL-2R; serum sIL-2R tracks CTCL burden, so surface and transcript diverge"
ADT_CD26,DPP4,-0.612,17,3.05e-05,True,0.000726,-1.87,0.0667,0.00638,,,
ADT_CD27,CD27,0.95,17,0.000214,True,0.00318,0.482,0.867,0.0195,,,
ADT_CD28,CD28,0.655,17,0.000153,True,0.00259,0.699,0.933,0.00638,,,
ADT_CD29,ITGB1,0.109,14,0.058,True,0.192,1.17,1,0.00638,,,
ADT_CD31,PECAM1,0.00343,17,0.782,False,0.878,-0.807,0,0.0265,,,
ADT_CD35,CR1,0.0249,17,0.174,False,0.296,-0.175,0.333,0.155,,,
ADT_CD38,CD38,-0.101,17,3.05e-05,False,0.000726,-0.554,0.0667,0.0251,CD38,phase 2/3,
ADT_CD39,ENTPD1,0.0443,17,0.00209,False,0.0178,1.06,0.8,0.0294,,,
ADT_CD43,SPN,0.0066,14,0.67,False,0.781,0.0256,0.533,1,,,
ADT_CD45RA,PTPRC,-0.214,17,0.0714,True,0.212,0.104,0.8,0.0998,,,
ADT_CD45RO,PTPRC,0.049,17,0.306,True,0.428,0.104,0.8,0.0998,,,
ADT_CD49a,ITGA1,-0.0538,17,0.174,False,0.296,-0.539,0.267,0.0729,,,
ADT_CD49d,ITGA4,-0.372,17,3.05e-05,False,0.000726,-2.91,0.0667,0.00638,,,
ADT_CD52,CD52,0.081,14,0.00403,True,0.0282,0.273,0.733,0.22,CD52,phase 2 (off-label),
ADT_CD55,CD55,-0.00629,14,0.583,False,0.715,0.634,0.733,0.0866,,,
ADT_CD56,NCAM1,0.0201,17,0.306,False,0.428,0,0.4,0.731,,,
ADT_CD56_QA17A16,NCAM1,0.00301,14,0.715,False,0.818,0,0.4,0.731,,,
ADT_CD62L,SELL,0.374,17,0.644,True,0.767,0.522,0.6,0.336,,,shed by ADAM17 within minutes of activation -- read SELL beside ADAM17
ADT_CD69,CD69,-0.0426,17,0.12,False,0.26,-0.64,0.2,0.0251,,,
ADT_CD70,CD70,0.03,17,0.174,False,0.296,1.73,0.933,0.0203,CD70,phase 1,
ADT_CD71,TFRC,0.0697,17,1.53e-05,False,0.000726,1.1,0.933,0.0142,,,
ADT_CD73,NT5E,-0.00612,17,0.517,False,0.662,0.452,0.667,0.264,,,
ADT_CD80,CD80,0.026,17,0.109,False,0.244,0.00821,0.667,0.322,,,
ADT_CD84,CD84,0.0318,14,0.0494,False,0.192,1.3,1,0.00638,CD84 / SLAMF5,phase 1/2,
ADT_CD86,CD86,0.0302,17,0.0395,False,0.174,0,0.333,0.259,,,
ADT_CD95,FAS,-0.047,17,0.159,False,0.292,-0.966,0.267,0.064,,,
ADT_CD101,CD101,-0.00438,17,0.207,False,0.32,0,0.333,0.471,,,
ADT_CD117,KIT,0.00565,17,0.89,False,0.927,-0.0437,0.267,0.134,,,
ADT_CD120b,TNFRSF1B,0.0698,14,0.058,False,0.192,0.847,0.533,0.31,TNFR2,phase 1/2a,
ADT_CD122,IL2RB,0.0257,17,0.284,False,0.412,-0.159,0.467,1,,,
ADT_CD124,IL4R,-0.0189,17,0.19,True,0.31,0.146,0.533,0.424,,,
ADT_CD126,IL6R,0.0299,14,0.058,False,0.192,0.541,0.933,0.00638,,,
ADT_CD127,IL7R,-0.228,17,0.0395,True,0.174,-0.404,0.2,0.0811,,,
ADT_CD134,TNFRSF4,-0.0371,17,0.378,False,0.523,0.391,0.667,0.189,,,
ADT_CD137,TNFRSF9,-0.00314,17,0.89,False,0.927,0.00736,0.533,0.247,,,
ADT_CD138,SDC1,0.0257,17,0.0348,False,0.173,0,0.133,0.325,,,
ADT_CD141,THBD,0.0215,17,0.12,False,0.26,0,0,1,,,
ADT_CD150,SLAMF1,0.0183,17,0.611,False,0.742,0.576,0.733,0.242,,,
ADT_CD152,CTLA4,0.026,17,0.0202,False,0.109,0.876,0.733,0.0574,,,constitutively internalised by AP2M1 via its YVKM motif; surface pool is a few % of total. LRBA/DEF6 recycling sets what is displayed
ADT_CD154,CD40LG,0.0388,17,0.89,False,0.927,-0.0617,0.467,0.731,,,
ADT_CD158b,KIR2DL3,0.0213,17,0.207,True,0.32,0.0316,0.533,0.703,,,
ADT_CD158e1,KIR3DL1,0.03,17,0.207,False,0.32,0.202,0.667,0.088,,,
ADT_CD161,KLRB1,-0.0142,17,0.431,False,0.582,-1.79,0.0667,0.0574,,,
ADT_CD163,CD163,0.0257,17,0.159,False,0.292,0,0.133,0.325,,,
ADT_CD183,CXCR3,0.0271,17,0.132,False,0.272,-1.33,0.2,0.0811,,,
ADT_CD184,CXCR4,0.0204,17,0.207,False,0.32,-0.669,0.0667,0.00638,,,
ADT_CD185,CXCR5,0.0286,17,0.0638,False,0.2,-0.362,0.0667,0.0294,,,
ADT_CD193,CCR3,0.0186,17,0.0887,False,0.24,0,0.133,0.62,,,
ADT_CD194,CCR4,0.823,17,0.00934,True,0.0585,1.19,0.8,0.121,CCR4,approved,ligand-induced internalisation (CCL17/CCL22); a high-CCL17 lesion lowers surface CCR4
ADT_CD195,CCR5,-0.0038,17,0.284,False,0.412,0,0.0667,0.114,,,
ADT_CD196,CCR6,-0.00038,17,0.712,False,0.818,-1.15,0.2,0.456,,,
ADT_CD197,CCR7,-0.0493,17,0.00557,True,0.0368,0.295,0.6,0.857,,,
ADT_CD223,LAG3,0.0344,17,0.0348,False,0.173,0.977,0.8,0.115,,,
ADT_CD244,CD244,0.00634,17,0.927,False,0.942,-0.13,0.133,0.0366,,,
ADT_CD274,CD274,0.0172,17,0.0984,False,0.244,0.195,0.733,0.222,,,
ADT_CD278,ICOS,-0.0928,17,0.644,True,0.767,-0.211,0.4,0.285,,,
ADT_CD279,PDCD1,0.269,17,0.000504,True,0.00599,1.22,0.933,0.00638,PD-1,phase 2,rapid internalisation and FBXO38-mediated degradation; transcript overstates surface
ADT_CD294,PTGDR2,0.0201,17,0.0505,False,0.192,0,0.333,0.731,,,
ADT_CD307c,FCRL3,0.0469,16,0.00336,False,0.0266,1.45,0.8,0.0392,,,
ADT_CD314,KLRK1,0.0225,17,0.854,False,0.925,-1.84,0.133,0.147,,,
ADT_CD319,SLAMF7,0.0196,17,0.0714,False,0.212,-0.115,0.0667,0.0698,,,
ADT_CD326,EPCAM,0.0352,17,0.00209,False,0.0178,0.0967,0.667,0.0438,,,
ADT_CD357,TNFRSF18,0.0244,17,0.579,False,0.715,0.232,0.6,0.456,,,
ADT_CD360,IL21R,0.0333,14,0.0785,False,0.226,0.437,0.733,0.22,,,
ADT_CD366,HAVCR2,0.0214,17,0.89,False,0.927,0.12,0.667,0.245,,,
ADT_CCR10,CCR10,0.0401,14,0.153,False,0.292,0.1,0.667,0.259,,,
ADT_CX3CR1,CX3CR1,0.023,17,0.459,False,0.613,0,0.333,0.771,,,
ADT_FceRIa,FCER1A,0.0182,17,0.243,False,0.367,0,0,0.456,,,
ADT_GARP,LRRC32,0.00594,17,0.487,False,0.644,0.15,0.733,0.0844,,,
ADT_GP130,IL6ST,0.0318,14,0.67,False,0.781,0.129,0.533,1,,,
ADT_GPR56,ADGRG1,0.0279,17,0.0448,False,0.19,0,0.4,0.906,,,
ADT_HLA-DR,HLA-DRA,-0.0574,17,0.015,False,0.085,-0.455,0.133,0.088,,,
ADT_KLRG1,KLRG1,0.0257,17,0.109,False,0.244,-0.808,0.133,0.022,,,
ADT_LAP,TGFB1,0.00317,14,0.903,False,0.927,-0.229,0.267,0.199,,,
ADT_TIGIT,TIGIT,0.0321,17,0.431,True,0.582,1.97,0.933,0.0195,,,
ADT_CD213a1,IL13RA1,0.0282,14,0.104,False,0.244,0,0.0667,0.456,,,
ADT_CD213a2,IL13RA2,0.023,14,0.135,False,0.273,0,0.0667,0.456,,,
ADT_CD217,IL17RA,0.0226,14,0.173,False,0.296,-0.15,0.2,0.242,,,
ADT_CLA,SELPLG,0.0273,14,0.0906,False,0.24,0.333,0.933,0.0195,,,
ADT_CD158k,KIR3DL2,0.0284,12,0.0522,False,0.192,2.73,1,0.00638,CD158k,phase 2,"stable surface receptor, internalises on ligand binding -- the basis of lacutamab ADCC"
```

## §3 Agreement rates

**What:** how often protein and transcript agree in direction.

**How:** Spearman and sign agreement, computed three ways. The middle row is the honest number; the first is what a version of this analysis that ignored the gate reported.

```csv
set,rho,same_direction,n_markers
every mapped marker,0.613,50,75
markers above isotype background,0.793,16,21
markers at background (meaningless),0.551,34,54
```

## §3 Markers where measured protein and transcript disagree

**What:** the candidates a transcript ranking would get wrong.

**How:** sign of the protein delta against sign of the transcript log2FC, measured markers only.

```csv
index,gene,cd_name,prot_delta,q_prot,log2fc_paired,q,n_donors_prot,surface_caveat
71,CCR7,,-0.0493,0.0368,0.295,0.857,17,
50,IL4R,,-0.0189,0.31,0.146,0.424,17,
8,ITGAL,,-0.0244,1,0.123,0.424,14,
29,PTPRC,,-0.214,0.212,0.104,0.0998,17,
5,CD5,CD5,0.229,0.272,-0.0778,0.643,17,
```

## §4a Variance partition, every measured marker

**What:** how much of the surface-protein variance the trafficking machinery explains after the marker's own mRNA.

**How:** incremental R2 on donor-demeaned single cells; `traffic_beats_own` is the flag that the transcript was never going to rank that marker correctly.

**Full table on disk:** `tables/tgt45_variance_partition.csv`

```csv
adt,gene,r2_umi,inc_r2_own_rna,inc_r2_traffic,r2_full,donor_share,traffic_beats_own,n_cells,prot_delta,log2fc_paired,detected,label
ADT_TIGIT,TIGIT,0.103,0.00116,0.0101,0.114,0.107,True,45993,0.0321,1.97,True,TIGIT
ADT_CD278,ICOS,0.0718,0.00453,0.00976,0.0861,0.15,True,45993,-0.0928,-0.211,True,ICOS
ADT_CD62L,SELL,0.256,0.105,0.00837,0.369,0.476,False,45993,0.374,0.522,True,SELL
ADT_CD194,CCR4,0.171,0.0364,0.00812,0.215,0.551,False,45993,0.823,1.19,True,CCR4
ADT_CD124,IL4R,0.122,7.72e-05,0.00639,0.129,0.0891,True,45993,-0.0189,0.146,True,IL4R
ADT_CD279,PDCD1,0.122,0.00757,0.00627,0.136,0.39,False,45993,0.269,1.22,True,PDCD1
ADT_CD197,CCR7,0.0933,0.000184,0.00435,0.0978,0.0636,True,45993,-0.0493,0.295,True,CCR7
ADT_CD45RO,PTPRC,0.0982,0.000745,0.00343,0.102,0.347,True,45993,0.049,0.104,True,PTPRC (CD45RO)
ADT_CD26,DPP4,0.0571,0.00256,0.00334,0.063,0.198,True,45993,-0.612,-1.87,True,DPP4
ADT_CD28,CD28,0.284,0.00317,0.00266,0.29,0.48,False,45993,0.655,0.699,True,CD28
ADT_CD5,CD5,0.166,0.00561,0.00184,0.173,0.442,False,45993,0.229,-0.0778,True,CD5
ADT_CD2,CD2,0.133,0.0207,0.00167,0.155,0.766,False,45993,-0.73,-0.802,True,CD2
ADT_CD27,CD27,0.358,0.00942,0.0016,0.369,0.182,False,45993,0.95,0.482,True,CD27
ADT_CD7,CD7,0.0416,0.114,0.00145,0.158,0.622,False,45993,-1.03,-1.78,True,CD7
ADT_CD45RA,PTPRC,0.00744,0.000131,0.00124,0.00881,0.254,True,45993,-0.214,0.104,True,PTPRC (CD45RA)
ADT_CD158b,KIR2DL3,0.113,0.00307,0.00116,0.117,0.0212,False,45993,0.0213,0.0316,True,KIR2DL3
ADT_CD127,IL7R,0.22,0.014,0.00108,0.235,0.338,False,45993,-0.228,-0.404,True,IL7R
ADT_CD3,CD3E,0.164,8.97e-07,0.000686,0.164,0.242,True,45993,-0.402,-0.173,True,CD3E
```

## §4e Held-out test against random programs

**What:** whether the machinery still helps on a donor it was not fitted on, and whether it beats size-matched random gene programs.

**How:** leave-one-donor-out gain in R2 over depth + own mRNA; Wilcoxon over donors (q_donor) and against 200 random-program draws (q_null). `beats_random` requires both.

**Full table on disk:** `tables/tgt45_machinery_test.csv`

```csv
label,gene,adt,gain_rna,gain_traffic,gain_traffic_q25,gain_traffic_q75,frac_donors_gain,p_donor,null_q05,null_median,null_q95,p_null,q_donor,q_null,beats_random
PTPRC (CD45RO),PTPRC,ADT_CD45RO,-0.000838,-0.00779,-0.0179,-6.44e-05,0.235,0.996,-0.013,-0.00509,-0.0011,0.786,0.996,0.924,False
CCR7,CCR7,ADT_CD197,0.000152,-0.0037,-0.0193,0.000944,0.294,0.99,-0.00331,-4.92e-06,0.00261,0.97,0.996,0.99,False
TIGIT,TIGIT,ADT_TIGIT,-0.000725,-0.00342,-0.0122,0.0061,0.412,0.836,-0.00525,-0.0014,0.00268,0.821,0.996,0.924,False
CD28,CD28,ADT_CD28,0.00254,-0.00294,-0.00422,0.000722,0.294,0.888,-0.00627,-0.00216,0.000959,0.682,0.996,0.924,False
IL4R,IL4R,ADT_CD124,7.32e-05,-0.00229,-0.0187,0.00167,0.412,0.956,-0.00169,0.00109,0.00446,0.99,0.996,0.99,False
PTPRC (CD45RA),PTPRC,ADT_CD45RA,0.000246,-0.00226,-0.00322,-4.71e-06,0.235,0.951,-0.018,-0.00597,-0.000809,0.129,0.996,0.776,False
CD7,CD7,ADT_CD7,0.102,-0.00201,-0.0132,0.000615,0.353,0.988,-0.0133,-0.00553,-0.00168,0.0697,0.996,0.776,False
CCR4,CCR4,ADT_CD194,0.0401,-0.0019,-0.00978,0.00743,0.471,0.627,-0.00579,2.95e-05,0.00638,0.726,0.996,0.924,False
KIR2DL3,KIR2DL3,ADT_CD158b,-0.000922,-0.00133,-0.00279,-0.000129,0.235,0.94,-0.00243,-0.000761,0.00115,0.741,0.996,0.924,False
IL7R,IL7R,ADT_CD127,0.00463,-0.00116,-0.00213,-9.83e-05,0.235,0.975,-0.00228,-0.000427,0.00265,0.771,0.996,0.924,False
ICOS,ICOS,ADT_CD278,0.00421,-0.00101,-0.00684,0.00979,0.412,0.591,-0.00416,-0.000495,0.00333,0.647,0.996,0.924,False
CD2,CD2,ADT_CD2,0.0128,-0.000512,-0.00278,0.00162,0.412,0.771,-0.00288,0.00016,0.00298,0.632,0.996,0.924,False
CD27,CD27,ADT_CD27,0.0112,-0.000405,-0.00214,0.001,0.353,0.771,-0.00302,-0.000265,0.00323,0.537,0.996,0.924,False
CD5,CD5,ADT_CD5,0.00317,3.45e-05,-0.0014,0.00272,0.529,0.518,-0.00251,0.000167,0.00309,0.557,0.996,0.924,False
CD3E,CD3E,ADT_CD3,-7.62e-05,0.000243,-0.00144,0.000998,0.588,0.573,-0.00122,-2.73e-05,0.00111,0.348,0.996,0.924,False
DPP4,DPP4,ADT_CD26,0.000477,0.000297,-0.00558,0.00181,0.588,0.71,-0.00137,0.00104,0.00518,0.652,0.996,0.924,False
PDCD1,PDCD1,ADT_CD279,0.00722,0.00155,-0.00975,0.00706,0.529,0.741,-0.00489,-0.00125,0.00219,0.114,0.996,0.776,False
SELL,SELL,ADT_CD62L,0.0664,0.00478,-0.00453,0.012,0.706,0.164,-0.00197,0.00236,0.00951,0.259,0.996,0.924,False
```

## §4b Best module per focus marker

**What:** which trafficking module explains most of each focus marker, and whether it is the one its biology predicts.

**How:** incremental R2 per module; `predicted` is a prior (IL2RA and TNFRSF1B are shed, CCR4 and KIR3DL2 internalise, CCR7 recycles), not a result.

**Full table on disk:** `tables/tgt45_module_partial_r2.csv`

```csv
gene,module,partial_r2,n_module_genes,expected,predicted,inc_r2_own_rna,traffic_beats_own
CCR4,ubiquitin_proteostasis,0.00463,4,False,endocytosis,0.0364,False
CCR7,ubiquitin_proteostasis,0.00253,4,False,recycling,0.000184,True
DPP4,ubiquitin_proteostasis,0.00189,4,False,,0.00256,True
CD28,ubiquitin_proteostasis,0.00128,4,False,,0.00317,False
CD5,er_folding_qc,0.000495,7,False,,0.00561,False
CD7,escrt_exosome,0.000425,4,False,er_folding_qc,0.114,False
```

## §4c Mixed model, focus markers

**What:** the formal version with donor as a random intercept.

**How:** protein ~ own mRNA + trafficking + depth, donor random intercept.

**Full table on disk:** `tables/tgt45_mixedlm_focus.csv`

```csv
gene,n_cells,b_rna,p_rna,donor_var,resid_var,top_modules
CD5,45993,0.0636,1.33e-67,0.401,0.379,"glycosylation +0.073*, recycling +0.061*, er_folding_qc -0.050*"
CD7,45993,0.338,0,0.751,0.394,"shedding -0.109*, escrt_exosome -0.053*, retrograde -0.045*"
DPP4,45993,0.274,7.39e-28,0,0.161,"snare_fusion +0.059*, ubiquitin_proteostasis +0.048*, shedding +0.029"
CD28,45993,0.0796,1.91e-49,0,0.748,"glycosylation -0.154*, ubiquitin_proteostasis -0.102*, snare_fusion -0.088*"
CCR4,45993,0.234,0,0.972,0.486,"snare_fusion -0.155*, ubiquitin_proteostasis -0.140*, glycosylation -0.118*"
CCR7,45993,-0.00571,0.000332,0,0.0884,"glycosylation +0.065*, snare_fusion +0.042*, ubiquitin_proteostasis +0.041*"
```

## §4d Donor-level replication

**What:** the same question on the 17 donor-level deltas, which is the quantity the screen ranks on.

**How:** saturated with 11 modules and 17 points -- the ordering survives, the value is an upper bound.

**Full table on disk:** `tables/tgt45_donor_level_traffic.csv`

```csv
index,gene,n_donors,rho_own_rna,p_own_rna,r2_own_rna,inc_r2_traffic
5,DPP4,17,0.282,0.273,0.00154,0.906
8,ITGB1,14,0.358,0.208,0.107,0.867
10,PTPRC,17,0.0931,0.722,0.00334,0.825
20,TIGIT,17,0.23,0.374,0.065,0.806
4,ITGAL,14,0.521,0.0562,0.182,0.781
15,KIR2DL3,17,0.0934,0.722,0.124,0.689
17,CCR7,17,-0.199,0.445,0.0135,0.684
13,IL4R,17,-0.0098,0.97,0.282,0.67
1,CD3E,17,0.105,0.687,0.00156,0.656
11,CD52,14,0.437,0.118,0.196,0.638
9,PTPRC,17,0.301,0.24,0.219,0.576
14,IL7R,17,0.681,0.0026,0.388,0.55
6,CD27,17,0.561,0.0191,0.409,0.498
18,ICOS,17,0.583,0.014,0.472,0.453
19,PDCD1,17,0.654,0.00437,0.541,0.375
0,CD2,17,0.77,0.000303,0.696,0.286
7,CD28,17,0.721,0.0011,0.665,0.201
2,CD5,17,0.826,4.41e-05,0.715,0.199
3,CD7,17,0.618,0.00824,0.56,0.191
16,CCR4,17,0.86,9.43e-06,0.805,0.122
12,SELL,17,0.912,3.51e-07,0.9,0.0869
```

======================== COPY TO HERE ==========================