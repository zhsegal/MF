> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
============================== BEGIN pseq29 numerical result (paste as one block) ==============================

# 29_perturbseq_reverse_engineering — MF malignant CD4 (skin) reverse-engineered with GWCD4i Stim48hr knockdowns, crossed with inferCNV

## Context
Cohort: 66 CD4-malignant MF skin donors (52 discovery / 14 replication), 286,748 CD4 cells (malignant vs same-donor reactive, call = mal_cluster). Signature = pydeseq2 Wald z, ~ donor + arm. Model = authors' pert2state (SVD D=60, elastic net, 5-fold x 3); X = 10,096 genes x 2,582 regulators. w>0: knockdown mimics MF; w<0: knockdown opposes MF.

## Model fit
model,n_genes,n_regulators,test_pearson,test_pearson_se,train_pearson
MF Stim48hr,10096,2582,0.611,0.004,0.619
scrambled,10096,2582,0.263,0.006,0.265


## Replication (held-out genes vs replication donors)
,split,n_test_genes,correlation,upper_correlation,pred_input_correlation
mean,7.0,2019.2,0.5871999999999999,0.8780666666666668,0.6115333333333334
std,4.47213595499958,0.41403933560541256,0.016823453017397214,0.005049280948336916,0.015495928804812249


## Control panel
group,regulator,coef_mean,coef_sem,coef_rank,abs_rank,sign_ok
LoF in CTCL (expect w>0),PTEN,-0.0032,0.0003,0.8896,581.0,False
LoF in CTCL (expect w>0),TNFAIP3,,,,,
LoF in CTCL (expect w>0),ZEB1,-0.0065,0.0002,0.9888,77.0,False
LoF in CTCL (expect w>0),CDKN2A,,,,,
LoF in CTCL (expect w>0),TP53,-0.0036,0.0002,0.912,461.0,False
LoF in CTCL (expect w>0),ARID1A,0.0001,0.0001,0.5025,2527.0,True
LoF in CTCL (expect w>0),DNMT3A,,,,,
LoF in CTCL (expect w>0),TET2,0.002,0.0001,0.2216,1119.0,True
LoF in CTCL (expect w>0),KMT2D,,,,,
LoF in CTCL (expect w>0),CD58,,,,,
LoF in CTCL (expect w>0),FAS,,,,,
LoF in CTCL (expect w>0),NFKBIA,,,,,
LoF in CTCL (expect w>0),RB1,,,,,
LoF in CTCL (expect w>0),CYLD,,,,,
LoF in CTCL (expect w>0),PRDM1,,,,,
GoF / dependency (expect w<0),PLCG1,-0.0077,0.0002,0.993,43.0,True
GoF / dependency (expect w<0),CARD11,-0.0034,0.0001,0.9031,511.0,True
GoF / dependency (expect w<0),STAT3,-0.001,0.0002,0.6559,1831.0,True
GoF / dependency (expect w<0),STAT5B,0.0,0.0001,0.5095,2566.0,False
GoF / dependency (expect w<0),JAK3,,,,,
GoF / dependency (expect w<0),JAK1,,,,,
GoF / dependency (expect w<0),CD28,-0.0003,0.0001,0.5548,2338.0,True
GoF / dependency (expect w<0),NFKB2,0.0023,0.0002,0.1906,971.0,False
GoF / dependency (expect w<0),MYC,0.003,0.0002,0.1286,649.0,False
GoF / dependency (expect w<0),TOX,,,,,
GoF / dependency (expect w<0),GATA3,0.0038,0.0002,0.0759,394.0,False
GoF / dependency (expect w<0),IRF4,-0.0004,0.0001,0.5684,2272.0,True
GoF / dependency (expect w<0),BATF,0.0016,0.0002,0.277,1410.0,False
GoF / dependency (expect w<0),TNFRSF1B,,,,,


## Top 25 w>0 (knockdown mimics MF)
regulator,coef_mean,coef_sem,cluster,arm,arm_class
UFM1,0.0114,0.0002,c41 protein glycosylation,chr13q,lost
SAP130,0.0113,0.0002,c4 unknown,chr2q,neutral
BRMS1,0.011,0.0003,c4 unknown,chr11q,neutral
DDRGK1,0.0105,0.0002,c41 protein glycosylation,chr20p,neutral
SIN3B,0.0101,0.0002,c6 unknown,chr19p,gained
HDAC1,0.0101,0.0002,c60 unknown,chr1p,neutral
BTAF1,0.0099,0.0002,c88 unknown,chr10q,lost
RO60,0.0098,0.0003,c2 Ino80 complex,chr1q,neutral
PAN3,0.0098,0.0002,c25 unknown,chr13q,lost
KMT2B,0.0097,0.0003,,chr19q,neutral
GCN1,0.009,0.0002,c77 unknown,chr12q,neutral
IRF1,0.0087,0.0003,,chr5q,lost
TPT1,0.0086,0.0002,c8 peptidyl-diphthamide metabolism,chr13q,lost
TYK2,0.0085,0.0002,,chr19p,gained
ACAD11,0.0085,1e-04,c41 protein glycosylation,chr3q,lost
UBA5,0.0084,0.0002,c41 protein glycosylation,chr3q,lost
ZC3H10,0.0084,0.0002,c80 unknown,chr12q,neutral
NFRKB,0.0084,0.0004,c2 Ino80 complex,chr11q,neutral
POLR2M,0.0083,0.0003,,chr15q,neutral
IMMP2L,0.0081,1e-04,c96 unknown,chr7q,gained
IRF9,0.0081,0.0004,,chr14q,neutral
UBAP2L,0.0079,0.0002,c76 EMC complex,chr1q,neutral
MED24,0.0079,0.0002,c10 mediator-SAGA,chr17q,gained
UFL1,0.0078,0.0002,c41 protein glycosylation,chr6q,lost
KAT7,0.0077,0.0002,c30 unknown,chr17q,gained


## Top 25 w<0 (knockdown opposes MF)
regulator,coef_mean,coef_sem,cluster,arm,arm_class
HNRNPH1,-0.007,0.0002,c106 unknown,chr5q,lost
XPO6,-0.0071,1e-04,,chr16p,gained
LAT,-0.0072,0.0002,c22 TCR signaling,chr16p,gained
CD3G,-0.0074,1e-04,c22 TCR signaling,chr11q,neutral
ZBED3,-0.0075,0.0002,c22 TCR signaling,chr5q,lost
TFDP1,-0.0077,0.0002,c42 unknown,chr13q,lost
PLCG1,-0.0077,0.0002,c22 TCR signaling,chr20q,lost
GP1BB,-0.008,1e-04,,chr22q,neutral
CD3D,-0.0081,1e-04,c22 TCR signaling,chr11q,neutral
HELLS,-0.0084,0.0002,,chr10q,lost
SMARCD3,-0.0085,0.0002,,chr7q,gained
RHOA,-0.0085,1e-04,,chr3p,neutral
INTS12,-0.0086,0.0003,c45 unknown,chr4q,lost
LARP4,-0.0087,0.0002,,chr12q,neutral
HEXIM1,-0.0088,0.0003,c58 ribosomal and mitochrondrial function,chr17q,gained
CCND3,-0.0089,1e-04,c63 unknown,chr6p,neutral
FITM2,-0.0093,0.0003,c21 TCR signaling,chr20q,lost
CD3E,-0.0097,0.0002,c22 TCR signaling,chr11q,neutral
PHF20,-0.0109,0.0003,,chr20q,lost
UBR5,-0.011,0.0002,c87 Fanconi anemia complex,chr8q,neutral
CTBP1,-0.0112,0.0003,c47 CtBP core complex,chr4p,neutral
ZAP70,-0.0123,0.0002,c22 TCR signaling,chr2q,neutral
RCOR1,-0.0126,0.0002,c47 CtBP core complex,chr14q,neutral
KDM1A,-0.0126,0.0003,c47 CtBP core complex,chr1p,neutral
LCP2,-0.0127,0.0002,c22 TCR signaling,chr5q,lost


## Pathway tests (q<0.1, any column)
set,n,mean_w,q_abs_greater,q_signed_greater,q_signed_less
c22 TCR signaling,11,-0.008,0.0,1.0,0.0
c2 Ino80 complex,11,0.0061,0.0,0.0,1.0
c32 Ragulator-Rag complex,11,-0.0037,0.0103,1.0,0.0
c7 nonsense-mediated decay,10,-0.0034,0.0141,1.0,0.0037
c31 Ragulator-Rag complex,12,-0.0032,0.0506,1.0,0.0001
c93 unknown,8,-0.0051,0.0506,1.0,0.0001
c36 citrate metabolism,8,0.0034,0.0787,0.0006,1.0
c64 unknown,17,-0.0023,0.0787,1.0,0.0051
c21 TCR signaling,100,-0.0012,0.2115,1.0,0.0
c19 NCOR/HDAC3 complex,6,0.0034,0.2825,0.0152,1.0
c54 unknown,33,0.0017,0.3122,0.003,1.0
c48 tRNA processing,9,-0.0023,0.3122,1.0,0.0109
c17 genome defense,5,-0.0032,0.3122,1.0,0.0109
c102 unknown,10,-0.0027,0.4907,1.0,0.0057
c51 Oxidative phosphorylation,22,-0.0012,0.4907,1.0,0.0666
c77 unknown,84,0.0017,0.5096,0.0,1.0
c99 TCR signaling,195,-0.001,0.588,1.0,0.0
c26 unknown,15,0.0013,0.597,0.0962,1.0
c27 unknown,89,0.0013,0.722,0.0001,1.0
c33 elongator complex,12,0.0025,0.722,0.0199,1.0
c68 tRNA processing,17,-0.0016,0.8319,1.0,0.0147
c106 unknown,54,-0.001,0.9784,1.0,0.0109
c9 mediator-SAGA,34,0.0012,0.9829,0.0307,1.0


## TCR-engagement axis
Spearman rho(w_MF, w_activation) = 0.278
program,signature,n_genes,pearson_y_vs_kd_sign,diff,p_perm
cluster 99 TCR signaling (Stim48hr),MF signature,6779.0,-0.371,-2.985,0.001
cluster 99 TCR signaling (Stim48hr),MF replication,6777.0,-0.371,-2.054,0.001
cluster 99 TCR signaling (Stim48hr),TCR activation (Arce),6784.0,-0.437,-5.298,0.001
cluster 21 TCR signaling (Stim8hr),MF signature,6685.0,-0.369,-3.147,0.001
cluster 21 TCR signaling (Stim8hr),MF replication,6684.0,-0.388,-2.275,0.001
cluster 21 TCR signaling (Stim8hr),TCR activation (Arce),6698.0,-0.37,-4.659,0.001
cluster 22 TCR signaling core (Stim8hr),MF signature,6424.0,-0.327,-2.742,0.001
cluster 22 TCR signaling core (Stim8hr),MF replication,6424.0,-0.333,-1.915,0.001
cluster 22 TCR signaling core (Stim8hr),TCR activation (Arce),6423.0,-0.387,-4.914,0.001


## 4A arm-class tests
test,n_set,n_neutral,mean_set,mean_neutral,cliffs_delta,p_mw,p_perm,expected
lost vs neutral,635,1395,1e-04,0.0,0.0095,0.3658,0.3866,greater
gained vs neutral,452,1395,1e-04,0.0,0.0102,0.6273,0.7493,less
"any event vs neutral (|w|, authors' test)",1087,1395,0.0022,0.0021,0.0126,0.2955,0.1439,greater


## 4B dosage-predicted program (per donor)
donor,class,n_arms,n_regulators,r_obs,null_mean,null_sd,p_null,study,split
D1__P171,lost,7,428,0.199,0.2,0.015,0.582,chennareddy2025,discovery
D1__P171,gained,8,457,0.198,0.203,0.014,0.672,chennareddy2025,discovery
Li2024_atlas__PT35,lost,8,354,0.205,0.186,0.02,0.194,li2024,discovery
Li2024_atlas__PT35,gained,7,591,0.205,0.198,0.014,0.338,li2024,discovery
Li2024_atlas__CTCL12,lost,7,379,0.348,0.305,0.028,0.05,li2024,discovery
Li2024_atlas__CTCL12,gained,8,543,0.304,0.318,0.024,0.736,li2024,discovery
Li2024_atlas__PT11,lost,7,345,0.244,0.226,0.021,0.159,li2024,discovery
Li2024_atlas__PT11,gained,8,621,0.217,0.242,0.014,0.965,li2024,discovery
Li2024_atlas__CTCL14,lost,8,493,0.317,0.323,0.027,0.577,li2024,replication
Li2024_atlas__CTCL14,gained,7,482,0.346,0.325,0.026,0.219,li2024,replication
D15__MF26,lost,6,522,0.112,0.075,0.017,0.03,il4ra2026,discovery
D15__MF26,gained,9,352,0.061,0.069,0.022,0.667,il4ra2026,discovery
Li2024_atlas__CTCL1,lost,8,514,0.173,0.151,0.02,0.144,li2024,discovery
Li2024_atlas__CTCL1,gained,7,354,0.141,0.146,0.022,0.637,li2024,discovery
D11__P5,lost,7,405,0.184,0.189,0.015,0.632,brentuximab2026,discovery
D11__P5,gained,8,424,0.193,0.189,0.016,0.463,brentuximab2026,discovery
D1__P303,lost,8,492,0.209,0.23,0.013,0.94,chennareddy2025,replication
D1__P303,gained,7,554,0.231,0.231,0.012,0.527,chennareddy2025,replication
D11__P3,lost,8,457,0.238,0.24,0.015,0.562,brentuximab2026,discovery
D11__P3,gained,7,425,0.214,0.237,0.015,0.935,brentuximab2026,discovery
D11__P6,lost,8,483,0.353,0.333,0.013,0.065,brentuximab2026,discovery
D11__P6,gained,7,493,0.328,0.332,0.015,0.647,brentuximab2026,discovery
D1__PGS,lost,6,379,0.237,0.201,0.017,0.01,chennareddy2025,discovery
D1__PGS,gained,9,706,0.22,0.216,0.011,0.348,chennareddy2025,discovery
D11__P1,lost,7,511,0.197,0.2,0.016,0.577,brentuximab2026,discovery
D11__P1,gained,8,430,0.206,0.195,0.016,0.234,brentuximab2026,discovery
D15__MF35,lost,8,435,0.368,0.354,0.016,0.174,il4ra2026,discovery
D15__MF35,gained,7,572,0.364,0.366,0.011,0.557,il4ra2026,discovery
D15__MF48,lost,8,500,0.287,0.272,0.013,0.139,il4ra2026,replication
D15__MF48,gained,7,355,0.271,0.259,0.016,0.239,il4ra2026,replication
D11__P2,lost,6,341,0.29,0.296,0.014,0.667,brentuximab2026,discovery
D11__P2,gained,9,586,0.326,0.319,0.01,0.254,brentuximab2026,discovery
Li2024_atlas__PT55,lost,8,428,0.029,0.027,0.014,0.413,li2024,discovery
Li2024_atlas__PT55,gained,7,585,0.017,0.027,0.012,0.766,li2024,discovery
Li2024_atlas__MF21,lost,7,437,0.343,0.323,0.015,0.1,li2024,replication
Li2024_atlas__MF21,gained,8,456,0.302,0.323,0.017,0.896,li2024,replication
Li2024_atlas__PT47,lost,9,631,0.295,0.28,0.015,0.164,li2024,discovery
Li2024_atlas__PT47,gained,6,424,0.256,0.269,0.021,0.721,li2024,discovery
Li2024_atlas__CTCL9,lost,8,393,0.363,0.355,0.019,0.303,li2024,discovery
Li2024_atlas__CTCL9,gained,7,520,0.372,0.369,0.015,0.448,li2024,discovery
Li2024_atlas__CTCL16,lost,9,508,0.394,0.369,0.02,0.109,li2024,discovery
Li2024_atlas__CTCL16,gained,6,421,0.35,0.362,0.022,0.701,li2024,discovery
D7__MF311,lost,7,405,0.156,0.16,0.02,0.592,rindler2021_mc,discovery
D7__MF311,gained,8,381,0.15,0.159,0.02,0.677,rindler2021_mc,discovery
D8__P163,lost,9,414,0.242,0.228,0.013,0.144,alkon2024,discovery
D8__P163,gained,6,443,0.242,0.232,0.013,0.219,alkon2024,discovery
Li2024_atlas__CTCL8,lost,7,350,0.146,0.159,0.013,0.851,li2024,replication
Li2024_atlas__CTCL8,gained,8,474,0.151,0.165,0.011,0.881,li2024,replication
Li2024_atlas__MF30,lost,9,533,0.279,0.299,0.014,0.891,li2024,replication
Li2024_atlas__MF30,gained,6,502,0.31,0.297,0.014,0.189,li2024,replication
D13__MF19,lost,5,270,0.217,0.217,0.025,0.552,gaydosik2022,discovery
D13__MF19,gained,7,456,0.245,0.234,0.02,0.323,gaydosik2022,discovery
Li2024_atlas__PT52,lost,8,495,0.297,0.281,0.024,0.279,li2024,discovery
Li2024_atlas__PT52,gained,7,499,0.281,0.279,0.024,0.542,li2024,discovery
Li2024_atlas__CTCL5,lost,9,649,0.275,0.28,0.011,0.687,li2024,discovery
Li2024_atlas__CTCL5,gained,6,405,0.257,0.267,0.014,0.756,li2024,discovery
Li2024_atlas__PT53,lost,10,631,0.256,0.253,0.012,0.458,li2024,discovery
Li2024_atlas__PT53,gained,5,399,0.207,0.241,0.014,0.985,li2024,discovery
Li2024_atlas__CTCL3,lost,9,431,0.312,0.269,0.019,0.005,li2024,discovery
Li2024_atlas__CTCL3,gained,6,400,0.234,0.267,0.019,0.94,li2024,discovery
Li2024_atlas__MF28,lost,9,457,0.296,0.3,0.018,0.612,li2024,discovery
Li2024_atlas__MF28,gained,6,500,0.294,0.305,0.016,0.731,li2024,discovery
D6__MF2,lost,7,407,0.25,0.26,0.022,0.721,gaydosik2019,replication
D6__MF2,gained,6,388,0.252,0.255,0.02,0.637,gaydosik2019,replication
D7__MF309,lost,9,596,0.115,0.107,0.012,0.284,rindler2021_mc,discovery
D7__MF309,gained,6,466,0.071,0.103,0.012,1.0,rindler2021_mc,discovery
Li2024_atlas__MF22,lost,9,511,0.17,0.222,0.015,1.0,li2024,replication
Li2024_atlas__MF22,gained,6,521,0.22,0.22,0.013,0.537,li2024,replication
Li2024_atlas__PT50,lost,6,327,0.396,0.36,0.038,0.154,li2024,replication
Li2024_atlas__PT50,gained,9,614,0.361,0.388,0.022,0.881,li2024,replication
Li2024_atlas__CTCL10,lost,7,360,0.261,0.268,0.016,0.697,li2024,discovery
Li2024_atlas__CTCL10,gained,8,597,0.282,0.283,0.011,0.522,li2024,discovery
D15__MF28,lost,6,505,0.075,0.071,0.017,0.413,il4ra2026,discovery
D15__MF28,gained,9,426,0.062,0.068,0.018,0.662,il4ra2026,discovery
D13__MF17,lost,8,375,0.298,0.247,0.027,0.01,gaydosik2022,discovery
D13__MF17,gained,6,366,0.235,0.244,0.025,0.627,gaydosik2022,discovery
D6__SC158,lost,6,334,0.267,0.25,0.03,0.318,gaydosik2019,discovery
D6__SC158,gained,8,480,0.27,0.265,0.022,0.423,gaydosik2019,discovery
D15__MF30,lost,9,516,0.257,0.29,0.028,0.891,il4ra2026,discovery
D15__MF30,gained,6,408,0.271,0.287,0.03,0.701,il4ra2026,discovery
D8__P90,lost,7,255,0.248,0.252,0.031,0.592,alkon2024,replication
D8__P90,gained,5,338,0.264,0.268,0.022,0.622,alkon2024,replication
D15__MF34,lost,7,547,0.246,0.206,0.022,0.045,il4ra2026,discovery
D15__MF34,gained,8,406,0.166,0.196,0.024,0.905,il4ra2026,discovery
Li2024_atlas__CTCL13,lost,7,365,0.175,0.176,0.016,0.542,li2024,discovery
Li2024_atlas__CTCL13,gained,8,597,0.161,0.188,0.013,0.98,li2024,discovery
D15__MF36,lost,7,477,0.413,0.409,0.015,0.453,il4ra2026,discovery
D15__MF36,gained,6,257,0.375,0.371,0.028,0.507,il4ra2026,discovery
D1__P312,lost,6,320,0.123,0.121,0.016,0.463,chennareddy2025,discovery
D1__P312,gained,9,436,0.132,0.128,0.012,0.368,chennareddy2025,discovery
Li2024_atlas__PT56,lost,8,369,-0.01,-0.024,0.023,0.249,li2024,discovery
Li2024_atlas__PT56,gained,7,529,-0.036,-0.027,0.018,0.697,li2024,discovery
D13__MF18,lost,7,300,0.141,0.126,0.016,0.154,gaydosik2022,replication
D13__MF18,gained,5,408,0.118,0.133,0.015,0.856,gaydosik2022,replication
Li2024_atlas__CTCL2,lost,4,220,0.173,0.184,0.022,0.706,li2024,discovery
Li2024_atlas__CTCL2,gained,5,381,0.227,0.202,0.017,0.06,li2024,discovery
D15__MF24,lost,4,239,0.197,0.19,0.02,0.383,il4ra2026,replication
D15__MF24,gained,6,330,0.187,0.2,0.018,0.791,il4ra2026,replication
Li2024_atlas__MF27,lost,10,541,0.439,0.462,0.018,0.886,li2024,discovery
Li2024_atlas__MF27,gained,5,363,0.421,0.438,0.023,0.781,li2024,discovery
D11__P4,lost,2,62,0.146,0.152,0.06,0.562,brentuximab2026,replication
D11__P4,gained,5,125,0.215,0.205,0.04,0.463,brentuximab2026,replication
D13__MF21,lost,8,455,0.356,0.378,0.027,0.836,gaydosik2022,discovery
D13__MF21,gained,7,415,0.411,0.375,0.028,0.075,gaydosik2022,discovery
D13__MF25,lost,3,156,0.16,0.155,0.025,0.453,gaydosik2022,discovery
D13__MF25,gained,4,388,0.181,0.18,0.015,0.498,gaydosik2022,discovery
D7__P73,lost,6,237,0.359,0.344,0.036,0.388,rindler2021_mc,discovery
D7__P73,gained,1,95,0.259,0.269,0.062,0.637,rindler2021_mc,discovery
D6__SC157,lost,6,307,0.203,0.198,0.02,0.408,gaydosik2019,discovery
D6__SC157,gained,4,263,0.207,0.194,0.022,0.289,gaydosik2019,discovery
D8__P138,lost,3,74,0.244,0.163,0.041,0.035,alkon2024,discovery
D8__P138,gained,1,45,0.186,0.133,0.041,0.139,alkon2024,discovery
D7__P84,lost,8,347,0.281,0.287,0.021,0.617,rindler2021_mc,replication
D7__P84,gained,2,221,0.278,0.261,0.029,0.313,rindler2021_mc,replication
Li2024_atlas__CTCL4,lost,3,102,0.11,0.108,0.029,0.453,li2024,discovery
Li2024_atlas__CTCL4,gained,3,204,0.148,0.129,0.023,0.224,li2024,discovery
Li2024_atlas__CTCL6,lost,1,42,0.138,0.124,0.043,0.413,li2024,discovery
Li2024_atlas__CTCL6,gained,3,204,0.211,0.205,0.028,0.403,li2024,discovery
D8__P65,lost,4,162,0.188,0.166,0.019,0.109,alkon2024,discovery
D8__P65,gained,1,24,0.056,0.086,0.032,0.826,alkon2024,discovery
D10__JONAK1,lost,6,232,0.327,0.328,0.025,0.562,jonak2021,discovery
D10__JONAK1,gained,2,74,0.244,0.236,0.05,0.478,jonak2021,discovery
D8__P150,lost,4,177,-0.034,-0.05,0.026,0.259,alkon2024,discovery
D1__P204,lost,4,99,0.298,0.277,0.053,0.393,chennareddy2025,discovery
D1__P204,gained,1,126,0.283,0.302,0.044,0.701,chennareddy2025,discovery
Li2024_atlas__CTCL7,lost,1,12,0.016,0.041,0.056,0.677,li2024,discovery
Li2024_atlas__CTCL7,gained,3,293,0.124,0.121,0.034,0.408,li2024,discovery


## 4C per-cell tests (regulator x arm x event class)
regulator,arm,event_class,n_donors,mean_rho,median_rho,stouffer_z,p_stouffer,q_stouffer,coef_mean,arm_class
UFL1,chr6q,lost,27,-0.1852,-0.1812,-inf,0.0,0.0,0.0078,lost
UFL1,chr6q,no event,32,-0.2047,-0.1678,-43.6918,0.0,0.0,0.0078,lost
UFM1,chr13q,lost,29,-0.1587,-0.1699,-42.3485,0.0,0.0,0.0114,lost
UBA5,chr3q,lost,19,-0.1296,-0.1013,-inf,0.0,0.0,0.0084,lost
ACAD11,chr3q,lost,19,-0.1317,-0.1182,-33.7591,0.0,0.0,0.0085,lost
UFL1,chr6q,gained,7,-0.3133,-0.3254,-30.5294,0.0,0.0,0.0078,lost
PLCG1,chr20q,no event,44,-0.0907,-0.0789,-30.4079,0.0,0.0,-0.0077,lost
DDRGK1,chr20p,no event,54,0.0851,0.086,30.2564,0.0,0.0,0.0105,neutral
ZAP70,chr2q,no event,57,0.083,0.0729,26.6803,0.0,0.0,-0.0123,neutral
PAN3,chr13q,lost,29,-0.1117,-0.0858,-26.6653,0.0,0.0,0.0098,lost
IMMP2L,chr7q,gained,38,0.0746,0.0742,25.7456,0.0,0.0,0.0081,gained
LCP2,chr5q,lost,19,0.143,0.1292,25.6028,0.0,0.0,-0.0127,lost
BTAF1,chr10q,lost,29,-0.1151,-0.093,-25.4015,0.0,0.0,0.0099,lost
KDM1A,chr1p,no event,60,-0.0671,-0.0663,-20.5824,0.0,0.0,-0.0126,neutral
UBA5,chr3q,no event,43,-0.0764,-0.0742,-20.165,0.0,0.0,0.0084,lost
BTAF1,chr10q,no event,36,-0.062,-0.0509,-18.0028,0.0,0.0,0.0099,lost
BRMS1,chr11q,no event,53,-0.0387,-0.0311,-17.1218,0.0,0.0,0.011,neutral
PAN3,chr13q,no event,33,-0.0719,-0.0732,-17.0208,0.0,0.0,0.0098,lost
ACAD11,chr3q,no event,43,-0.0631,-0.0583,-16.5628,0.0,0.0,0.0085,lost
INTS12,chr4q,lost,22,0.0712,0.0839,16.3132,0.0,0.0,-0.0086,lost
IRF1,chr5q,no event,42,0.0533,0.0432,15.7106,0.0,0.0,0.0087,lost
UFM1,chr13q,no event,33,-0.0794,-0.07,-15.5107,0.0,0.0,0.0114,lost
SIN3B,chr19p,gained,25,-0.0615,-0.0647,-15.4695,0.0,0.0,0.0101,gained
ZAP70,chr2q,gained,6,0.1615,0.1842,15.4403,0.0,0.0,-0.0123,neutral
LCP2,chr5q,no event,42,0.0781,0.0845,14.5042,0.0,0.0,-0.0127,lost
SAP130,chr2q,no event,57,-0.0424,-0.0441,-13.8643,0.0,0.0,0.0113,neutral
FITM2,chr20q,no event,44,-0.035,-0.0275,-12.9475,0.0,0.0,-0.0093,lost
PLCG1,chr20q,lost,20,-0.1209,-0.0916,-12.7925,0.0,0.0,-0.0077,lost
TPT1,chr13q,lost,29,-0.0505,-0.0534,-12.6485,0.0,0.0,0.0086,lost
KDM1A,chr1p,gained,2,-0.1739,-0.1739,-11.6704,0.0,0.0,-0.0126,neutral
RCOR1,chr14q,gained,1,0.1118,0.1118,11.429,0.0,0.0,-0.0126,neutral
ZAP70,chr2q,lost,3,0.1619,0.1504,11.357,0.0,0.0,-0.0123,neutral
CTBP1,chr4p,no event,51,0.0382,0.037,10.5065,0.0,0.0,-0.0112,neutral
BRMS1,chr11q,gained,5,-0.0963,-0.0687,-9.387,0.0,0.0,0.011,neutral
INTS12,chr4q,no event,38,0.0532,0.0506,9.1236,0.0,0.0,-0.0086,lost
PLCG1,chr20q,gained,2,-0.1017,-0.1017,-9.0798,0.0,0.0,-0.0077,lost
RCOR1,chr14q,no event,59,0.0218,0.019,8.7853,0.0,0.0,-0.0126,neutral
FITM2,chr20q,lost,20,-0.09,-0.0506,-7.9385,0.0,0.0,-0.0093,lost
SIN3B,chr19p,no event,35,-0.0384,-0.0249,-7.6548,0.0,0.0,0.0101,gained
SAP130,chr2q,gained,6,-0.0621,-0.0298,-7.4833,0.0,0.0,0.0113,neutral
FITM2,chr20q,gained,2,-0.0601,-0.0601,-7.4483,0.0,0.0,-0.0093,lost
SAP130,chr2q,lost,3,-0.1001,-0.0691,-7.3145,0.0,0.0,0.0113,neutral
HELLS,chr10q,lost,29,0.0249,0.0322,7.0527,0.0,0.0,-0.0084,lost
MED24,chr17q,gained,29,0.0317,0.0393,7.0422,0.0,0.0,0.0079,gained
UFM1,chr13q,gained,4,-0.2071,-0.2497,-6.839,0.0,0.0,0.0114,lost
HEXIM1,chr17q,no event,37,0.0251,0.0214,6.351,0.0,0.0,-0.0088,gained
IMMP2L,chr7q,no event,28,0.048,0.0255,6.1372,0.0,0.0,0.0081,gained
TYK2,chr19p,no event,35,-0.017,-0.0344,-6.1333,0.0,0.0,0.0085,gained
HEXIM1,chr17q,gained,29,-0.02,-0.0272,-5.6417,0.0,0.0,-0.0088,gained
MED24,chr17q,no event,37,-0.0144,-0.0109,-5.443,0.0,0.0,0.0079,gained
DDRGK1,chr20p,lost,7,0.0482,0.0304,5.3979,0.0,0.0,0.0105,neutral
BRMS1,chr11q,lost,8,-0.0484,-0.0566,-5.0328,0.0,0.0,0.011,neutral
TPT1,chr13q,no event,33,-0.0084,-0.0077,-4.9608,0.0,0.0,0.0086,lost
UBA5,chr3q,gained,4,0.0156,0.0079,4.8475,0.0,0.0,0.0084,lost
TYK2,chr19p,lost,6,-0.0798,-0.0732,-4.8255,0.0,0.0,0.0085,gained
IRF1,chr5q,lost,19,0.0326,0.014,4.2324,0.0,0.0,0.0087,lost
PHF20,chr20q,no event,44,-0.0194,-0.0086,-3.9663,0.0001,0.0001,-0.0109,lost
PHF20,chr20q,gained,2,0.0169,0.0169,3.6179,0.0003,0.0004,-0.0109,lost
DDRGK1,chr20p,gained,5,0.1859,0.1565,3.5958,0.0003,0.0004,0.0105,neutral
PHF20,chr20q,lost,20,-0.0309,-0.0147,-3.3524,0.0008,0.001,-0.0109,lost
IRF1,chr5q,gained,5,0.0146,-0.0002,3.2607,0.0011,0.0014,0.0087,lost
TYK2,chr19p,gained,25,0.0113,0.0229,3.09,0.002,0.0025,0.0085,gained
CTBP1,chr4p,gained,15,0.0329,0.0135,3.0271,0.0025,0.003,-0.0112,neutral
LCP2,chr5q,gained,5,-0.0111,0.0066,-2.7909,0.0053,0.0062,-0.0127,lost
SIN3B,chr19p,lost,6,-0.045,-0.0365,-2.4055,0.0161,0.0189,0.0101,gained
SMARCD3,chr7q,no event,28,-0.0014,0.0291,1.9419,0.0522,0.0601,-0.0085,gained
PAN3,chr13q,gained,4,-0.0678,-0.1052,-1.8979,0.0577,0.0649,0.0098,lost
HELLS,chr10q,no event,36,0.009,0.0144,1.8954,0.058,0.0649,-0.0084,lost
ACAD11,chr3q,gained,4,-0.0127,0.0221,1.7962,0.0725,0.0798,0.0085,lost
BTAF1,chr10q,gained,1,-0.1584,-0.1584,-1.7498,0.0802,0.087,0.0099,lost
SMARCD3,chr7q,gained,38,-0.0055,0.0031,-1.3727,0.1699,0.1798,-0.0085,gained
TPT1,chr13q,gained,4,0.0541,0.0623,1.3711,0.1703,0.1798,0.0086,lost
RCOR1,chr14q,lost,6,0.0111,0.0099,0.992,0.3212,0.3344,-0.0126,neutral
KDM1A,chr1p,lost,4,0.0155,0.0177,0.6401,0.5221,0.5362,-0.0126,neutral
HELLS,chr10q,gained,1,-0.0238,-0.0238,-0.2608,0.7942,0.8048,-0.0084,lost
INTS12,chr4q,gained,6,0.0366,0.0174,-0.0736,0.9413,0.9413,-0.0086,lost


============================== END pseq29 ==============================