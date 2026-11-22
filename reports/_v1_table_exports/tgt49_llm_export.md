> **Annotation vintage: v1.** Built on `cell_type_final == "CD4"`, whose benign CD4 arm is only
> ~46% conventional CD4 in skin — 16% CD8, 11% Treg, 8% NK/ILC, 15% unresolved
> (`reports/lineage_check/`). Statements about the malignant side stand; statements about the
> *benign* side, and any gene that separates CD4 from CD8/NK/Treg, are confounded. The current
> lineage call is `data/atlas_joint/nonmal_T_labels_v1.parquet`, read via `helpers/lineage.py`.
> See `KNOWLEDGE/RESULTS.md`.
======================== COPY FROM HERE ========================

# 49_skin_cd4_surfaceome_numeric — Skin CD4 surfaceome: malignant vs reactive CD4

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**, CD4 T cells only. Source object skin_T_tcr_annotated_v4.h5ad
(539,916 T/NK cells, 42,347 genes, raw counts), aggregated per
(arm, donor, gene) by jobs/run_target_screen_qc.py. Expression = log1p CP10K on the true library
size (obs.total_counts); `mean_all` = mean over a donor-arm's cells, `pct_expr` = fraction of
cells with >= 1 UMI. A donor-arm enters with >= 20 cells.

Arms: malignancy call `mal_cluster` (per-cell, made at the level of per-sample Leiden
clusters). CD4_malignant = CD4 cells called malignant; CD4_reactive = CD4 cells called
non-malignant in a patient; CD4_HC = CD4 cells of healthy-skin donors. CD8, Treg and
CD4_unassessed are not analysed.

Cohort: 146 donor keys -> 109 independent keys
(107 people) after dropping non-CD4 clones,
off-spectrum diagnoses (LyP/SPP/LPP) and repeat biopsies. 75 donors have both a
malignant and a reactive CD4 arm ("paired donors"); 21 healthy donors carry CD4_HC.

Genes: 2,115 tested surface genes (CTCL panel UNION confident surfaceome), plus
lineage-control (non-T) and trafficking-machinery reference genes.

Headline test: per gene, paired over donors. log2fc_paired = median over donors of
log2((mean_all_malig + 0.05) / (mean_all_react + 0.05)), a ratio of log-scale means and so
compressed relative to a raw fold change; Wilcoxon signed-rank on per-donor differences; BH ->
q. The same on pct_expr -> *_detect. Malignant vs healthy CD4 is unpaired (Mann-Whitney on
donor-level mean_all) -> *_vs_hc.

Shortlist gates: surface gene, not a lineage control or loss marker, q < 0.05,
log2fc_paired > 0.5, >= 10% of malignant cells positive in >= 50%
of patients. No artifact filter is run here; the companion notebook's six artifact tests
excluded nothing on this build because its contamination null was not calibrated. Genes flagged
`disputed_in_42` passed the statistics but are doubtful surface proteins by localisation.
Transcript only: skin has no surface protein measurement.

Calibration: CCR4, TNFRSF8 (CD30), KIR3DL2 and IL2RA are approved or registrational CTCL
antigens and should rank high; CD7 and DPP4 (CD26) loss is the CTCL phenotype and should be
negative.

## §1a Input

**What:** the object, call, normalisation and gene counts behind every table below.

**How:** read from the builder manifest, data/targets/skin_target_manifest.json.

```csv
key,value
object,skin_T_tcr_annotated_v4.h5ad
tables_built,2026-10-26 13:59
cells_in_object,539916
genes_in_object,42347
genes_screened_total,2355
genes_surface_panel_or_surfaceome,2115
genes_lineage_controls_added,30
genes_trafficking_machinery_added,210
panel_genes_absent_from_object,KIR2DL2
malignancy_call,mal_cluster
library_size_for_cp10k,obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg
min_cells_per_donor_arm,20
```

## §1b Cohort summary

**What:** how many donor keys, patients and paired donors the analysis rests on.

**How:** skin_donor_table() drop rules (non-CD4 clone, off-spectrum dx, repeat biopsy), then arm presence in the CD4 tables (>= 20 cells per donor-arm).

```csv
key,value
donor_keys_in_object,146
dropped: malignant clone is not CD4 (gd),7
dropped: diagnosis outside the MF/SS spectrum (LyP),7
dropped: diagnosis outside the MF/SS spectrum (SPP),6
dropped: malignant clone is not CD4 (CD8),6
dropped: malignant clone is not CD4 (DN),2
dropped: same person as D7__MF311,2
dropped: diagnosis outside the MF/SS spectrum (LPP),2
dropped: same person as D7__MF309,1
dropped: same person as D8__P90,1
dropped: same person as D8__P65,1
dropped: same person as D1__P312,1
dropped: same person as D13__MF19,1
independent_donor_keys,109
people,107
donors_with_malignant_arm,78
donors_with_reactive_arm,82
donors_with_healthy_CD4_arm,21
paired_donors (malignant AND reactive),75
paired_people,75
"paired donors, dx = MF",67
"paired donors, dx = SS",6
"paired donors, dx = MF/SS",2
```

## §1c Dropped donor keys

**What:** every donor key removed from the cohort, with the reason.

**How:** curated clin_* columns only; n_cells = all T/NK cells of the key in the object.

```csv
donor,study,dx,phenotype,lesion,person,n_cells,drop_reason
D8__P143,alkon2024,LPP,not_applicable,patch,vienna:P143,859,diagnosis outside the MF/SS spectrum (LPP)
D8__P194,alkon2024,LPP,not_applicable,patch,vienna:P194,2166,diagnosis outside the MF/SS spectrum (LPP)
D9__P101,lyp2026,LyP,CD4,LyP_lesion,vienna:P101,3509,diagnosis outside the MF/SS spectrum (LyP)
D9__P118,lyp2026,LyP,CD4,LyP_lesion,vienna:P118,892,diagnosis outside the MF/SS spectrum (LyP)
D9__P193,lyp2026,LyP,gd,LyP_lesion,vienna:P193,11896,diagnosis outside the MF/SS spectrum (LyP)
D9__P213,lyp2026,LyP,CD8,LyP_lesion,vienna:P213,7308,diagnosis outside the MF/SS spectrum (LyP)
D9__P217,lyp2026,LyP,CD4,LyP_lesion,vienna:P217,625,diagnosis outside the MF/SS spectrum (LyP)
D9__P220,lyp2026,LyP,CD8,LyP_lesion,vienna:GSM9038965,5259,diagnosis outside the MF/SS spectrum (LyP)
D9__PEN,lyp2026,LyP,CD8,LyP_lesion,vienna:PEN,4739,diagnosis outside the MF/SS spectrum (LyP)
D8__P102,alkon2024,SPP,not_applicable,patch,vienna:P102,1384,diagnosis outside the MF/SS spectrum (SPP)
D8__P144,alkon2024,SPP,not_applicable,patch,vienna:P144,106,diagnosis outside the MF/SS spectrum (SPP)
D8__P149,alkon2024,SPP,not_applicable,patch,vienna:P149,654,diagnosis outside the MF/SS spectrum (SPP)
D8__P78,alkon2024,SPP,not_applicable,patch,vienna:P78,417,diagnosis outside the MF/SS spectrum (SPP)
D8__P79,alkon2024,SPP,not_applicable,patch,vienna:P79,1645,diagnosis outside the MF/SS spectrum (SPP)
D8__P88,alkon2024,SPP,not_applicable,patch,vienna:P88,865,diagnosis outside the MF/SS spectrum (SPP)
D1__P195,chennareddy2025,pcAECTCL,CD8,plaque,vienna:P195,11243,malignant clone is not CD4 (CD8)
D1__P218,chennareddy2025,pcAECTCL,CD8,plaque,vienna:P195,2547,malignant clone is not CD4 (CD8)
D1__P230,chennareddy2025,pcAECTCL,CD8,tumor,vienna:P230,694,malignant clone is not CD4 (CD8)
D1__P95,chennareddy2025,pcAECTCL,CD8,tumor,vienna:P95,5843,malignant clone is not CD4 (CD8)
D6__MF5,gaydosik2019,CD8_aggr_CTCL,CD8,tumor,pittsburgh:SC82,5350,malignant clone is not CD4 (CD8)
Li2024_atlas__MF14,li2024,MF,CD8,plaque,mda:MF14,4369,malignant clone is not CD4 (CD8)
D8__P107,alkon2024,MF,DN,patch,vienna:P107,2020,malignant clone is not CD4 (DN)
D6__SC205,gaydosik2019,MF,DN,tumor,pittsburgh:SC205,1486,malignant clone is not CD4 (DN)
D1__P124,chennareddy2025,MF,gd,plaque,vienna:MF318,3158,malignant clone is not CD4 (gd)
D1__P127,chennareddy2025,MF,gd,plaque,vienna:P127,1772,malignant clone is not CD4 (gd)
D1__P202,chennareddy2025,MF,gd,plaque,vienna:P202,8040,malignant clone is not CD4 (gd)
D1__P318_thick,chennareddy2025,MF,gd,plaque,vienna:MF318,3759,malignant clone is not CD4 (gd)
D1__P318_thin,chennareddy2025,MF,gd,patch,vienna:MF318,3683,malignant clone is not CD4 (gd)
D1__P71,chennareddy2025,MF,gd,tumor,vienna:P71,548,malignant clone is not CD4 (gd)
D1__P97,chennareddy2025,MF,gd,plaque,vienna:MF318,4019,malignant clone is not CD4 (gd)
H__SS2,herrera2021,SS,CD4,erythroderma,pittsburgh:MF19,191,same person as D13__MF19
D7__MF312,rindler2021_mc,MF,CD4,patch,vienna:MF312,499,same person as D1__P312
D1__P76,chennareddy2025,MF,CD4,tumor,vienna:MF309,7018,same person as D7__MF309
D1__P311,chennareddy2025,MF,CD4,plaque,vienna:MF311,5258,same person as D7__MF311
D9__P72,lyp2026,MF,CD4,erythroderma,vienna:MF311,2671,same person as D7__MF311
D7__P65,rindler2021_mc,MF,CD4,nonlesional,vienna:P65,1927,same person as D8__P65
D7__P90,rindler2021_mc,MF,CD4,nonlesional,vienna:P90,129,same person as D8__P90
```

## §1d Cohort, one row per independent donor key

**What:** the clinical description and CD4 composition of every donor that can enter a table.

**How:** clinical fields from clin_*; n_malignant / n_reactive = CD4 cells in each arm (NaN = arm absent or < 20 cells); burden = n_malignant / (n_malignant + n_reactive); paired = has both arms; cd4_arms = the CD4 arms present.

**Full table on disk:** `tables/tgt49_cohort.csv`

```csv
donor,study,dx,dx2,phenotype,lesion,lesion_class,clin_stage,clin_stage_class,sex,age,is_cultured,n_cells_object,n_malignant,n_reactive,burden,paired,cd4_arms
D1__P112,chennareddy2025,HC,HC,not_applicable,healthy_skin,healthy,,,F,51,False,269,,,,False,CD4_HC
D1__P115,chennareddy2025,HC,HC,not_applicable,healthy_skin,healthy,,,M,48,False,605,,,,False,CD4_HC
D1__P116,chennareddy2025,HC,HC,not_applicable,healthy_skin,healthy,,,F,56,False,485,,,,False,CD4_HC
D1__P121,chennareddy2025,HC,HC,not_applicable,healthy_skin,healthy,,,F,44,False,83,,,,False,CD4_HC
D6__NOR1,gaydosik2019,HC,HC,not_applicable,healthy_skin,healthy,,,M,64,False,62,,,,False,CD4_HC
D6__NOR2,gaydosik2019,HC,HC,not_applicable,healthy_skin,healthy,,,F,48,False,66,,,,False,CD4_HC
D6__NOR3,gaydosik2019,HC,HC,not_applicable,healthy_skin,healthy,,,M,54,False,97,,,,False,CD4_HC
D6__NOR4,gaydosik2019,HC,HC,not_applicable,healthy_skin,healthy,,,M,61,False,73,,,,False,CD4_HC
D13__SC327_HTO1,gaydosik2022,HC,HC,not_applicable,healthy_skin,healthy,,,,,False,28,,,,False,CD4_HC
D13__SC327_HTO2,gaydosik2022,HC,HC,not_applicable,healthy_skin,healthy,,,,,False,225,,,,False,CD4_HC
D13__SC374_HTO10,gaydosik2022,HC,HC,not_applicable,healthy_skin,healthy,,,,,False,935,,,,False,CD4_HC
D13__SC374_HTO9,gaydosik2022,HC,HC,not_applicable,healthy_skin,healthy,,,,,False,230,,,,False,CD4_HC
D14__SC320,gaydosik2023,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,140,,,,False,CD4_HC
D14__SC348,gaydosik2023,HC,HC,not_applicable,healthy_skin,healthy,,,,,False,494,,,,False,CD4_HC
D14__SC353,gaydosik2023,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,253,,,,False,CD4_HC
D14__SC354,gaydosik2023,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,111,,,,False,CD4_HC
D14__SC405,gaydosik2023,HC,HC,not_applicable,healthy_skin,healthy,,,M,,False,149,,,,False,CD4_HC
H__HC1,herrera2021,HC,HC,not_applicable,healthy_skin,healthy,,,M,41,False,25,,,,False,
D15__NS13,il4ra2026,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,390,,,,False,CD4_HC
D15__NS14,il4ra2026,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,554,,,,False,CD4_HC
D15__NS16,il4ra2026,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,91,,,,False,CD4_HC
D15__NS17,il4ra2026,HC,HC,not_applicable,healthy_skin,healthy,,,F,,False,352,,,,False,CD4_HC
D8__P138,alkon2024,MF,MF,CD4,patch,lesion,IA,early,M,47,False,1489,627,614,0.505,True,CD4_malignant|CD4_reactive
D8__P150,alkon2024,MF,MF,,patch,lesion,IA,early,M,56,False,4061,2393,912,0.724,True,CD4_malignant|CD4_reactive
D8__P163,alkon2024,MF,MF,,patch,lesion,IB,early,F,83,False,1482,881,417,0.679,True,CD4_malignant|CD4_reactive
D8__P65,alkon2024,MF,MF,CD4,patch,lesion,IB,early,M,53,False,1386,615,626,0.496,True,CD4_malignant|CD4_reactive
D8__P90,alkon2024,MF,MF,CD4,patch,lesion,IB,early,M,75,False,393,68,232,0.227,True,CD4_malignant|CD4_reactive
D11__P1,brentuximab2026,MF,MF,CD4,plaque,lesion,IIIA,advanced,F,52,False,898,678,90,0.883,True,CD4_malignant|CD4_reactive
D11__P2,brentuximab2026,MF,MF,CD4,tumor,lesion,IIIA,advanced,F,48,False,11850,9473,1063,0.899,True,CD4_malignant|CD4_reactive
D11__P3,brentuximab2026,MF,MF,CD4,tumor,lesion,IIB,advanced,F,55,False,10804,7596,1589,0.827,True,CD4_malignant|CD4_reactive
D11__P4,brentuximab2026,MF,MF,CD4,plaque,lesion,IIIA,advanced,F,45,False,897,174,512,0.254,True,CD4_malignant|CD4_reactive
D11__P5,brentuximab2026,MF,MF,CD4,tumor,lesion,IVA2,advanced,F,61,False,9000,6550,1370,0.827,True,CD4_malignant|CD4_reactive
D11__P6,brentuximab2026,MF,MF,CD4,plaque,lesion,IVA2,advanced,M,64,False,7515,4575,1376,0.769,True,CD4_malignant|CD4_reactive
D3__P105,brunner2024,MF,MF,CD4,erythroderma,lesion,IVA1,advanced,M,84,False,715,117,516,0.185,True,CD4_malignant|CD4_reactive
D3__P196,brunner2024,MF,MF,CD4,erythroderma,lesion,IVA1,advanced,F,75,False,797,506,134,0.791,True,CD4_malignant|CD4_reactive
D3__P94,brunner2024,MF,MF,CD4,erythroderma,lesion,IVA1,advanced,M,71,False,3042,,2701,,False,CD4_reactive
D1__P171,chennareddy2025,MF,MF,CD4,tumor,lesion,IVA1,advanced,M,84,False,6288,5216,382,0.932,True,CD4_malignant|CD4_reactive
D1__P204,chennareddy2025,MF,MF,CD4,tumor,lesion,IIB,advanced,F,71,False,11668,3444,5741,0.375,True,CD4_malignant|CD4_reactive
D1__P303,chennareddy2025,MF,MF,CD4,tumor,lesion,IVB,advanced,F,65,False,1524,1038,116,0.899,True,CD4_malignant|CD4_reactive
D1__P312,chennareddy2025,MF,MF,CD4,plaque,lesion,IIB,advanced,M,55,False,1982,717,550,0.566,True,CD4_malignant|CD4_reactive
D1__PGS,chennareddy2025,MF,MF,CD4,erythroderma,lesion,IVA1,advanced,M,75,False,3193,2056,459,0.817,True,CD4_malignant|CD4_reactive
D6__SC157,gaydosik2019,MF,MF,CD4,tumor,lesion,IIB,advanced,F,83,False,1730,791,593,0.572,True,CD4_malignant|CD4_reactive
D6__SC158,gaydosik2019,MF,MF,CD4,plaque,lesion,IIB,advanced,F,46,False,461,306,99,0.756,True,CD4_malignant|CD4_reactive
D13__MF17,gaydosik2022,MF,MF,CD4,tumor,lesion,IIB,advanced,F,76,False,3991,3696,56,0.985,True,CD4_malignant|CD4_reactive
D13__MF18,gaydosik2022,MF,MF,CD4,tumor,lesion,IIB,advanced,M,87,False,2677,826,935,0.469,True,CD4_malignant|CD4_reactive
D13__MF21,gaydosik2022,MF,MF,CD4,tumor,lesion,IIB,advanced,M,61,False,966,118,547,0.177,True,CD4_malignant|CD4_reactive
H__MFIV1,herrera2021,MF,MF,CD4,plaque,lesion,IVB,advanced,F,39,False,98,44,27,0.62,True,CD4_malignant|CD4_reactive
D15__MF22,il4ra2026,MF,MF,,tumor,lesion,IIB,advanced,M,80,True,31,,,,False,
D15__MF24,il4ra2026,MF,MF,CD4,tumor,lesion,IIB,advanced,M,42,False,3635,643,1958,0.247,True,CD4_malignant|CD4_reactive
D15__MF26,il4ra2026,MF,MF,,erythroderma,lesion,IVB,advanced,M,30,False,3599,3041,265,0.92,True,CD4_malignant|CD4_reactive
D15__MF28,il4ra2026,MF,MF,,tumor,lesion,IIB,advanced,F,74,True,1195,123,691,0.151,True,CD4_malignant|CD4_reactive
D15__MF30,il4ra2026,MF,MF,,erythroderma,lesion,IIIB,advanced,M,80,True,1047,497,354,0.584,True,CD4_malignant|CD4_reactive
D15__MF34,il4ra2026,MF,MF,,tumor,lesion,IIB,advanced,M,76,True,1113,248,692,0.264,True,CD4_malignant|CD4_reactive
D15__MF35,il4ra2026,MF,MF,,mixed,lesion,IIB,advanced,F,66,True,4182,2187,597,0.786,True,CD4_malignant|CD4_reactive
D15__MF36,il4ra2026,MF,MF,,mixed,lesion,IIB,advanced,M,48,False,4311,620,2182,0.221,True,CD4_malignant|CD4_reactive
D15__MF40,il4ra2026,MF,MF,,erythroderma,lesion,IIIB,advanced,M,75,True,818,,576,,False,CD4_reactive
D15__MF48,il4ra2026,MF,MF,,papule,lesion,IIB,advanced,M,71,False,4681,3650,709,0.837,True,CD4_malignant|CD4_reactive
D10__JONAK1,jonak2021,MF,MF,CD4,plaque,lesion,IB,early,F,33,False,2799,348,1920,0.153,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL1,li2024,MF,MF,,mixed,lesion,IIB,advanced,F,38,False,9232,7676,352,0.956,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL10,li2024,MF,MF,,skin_unspecified,unspecified,IB,early,F,56,False,1429,1151,157,0.88,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL11,li2024,MF,MF,,skin_unspecified,unspecified,IB,early,F,58,False,102,90,,,False,CD4_malignant
Li2024_atlas__CTCL12,li2024,MF,MF,,skin_unspecified,unspecified,IV,advanced,M,60,False,3491,2784,331,0.894,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL13,li2024,MF,MF,,skin_unspecified,unspecified,IIB,advanced,M,68,False,11240,5962,2884,0.674,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL14,li2024,MF,MF,,skin_unspecified,unspecified,IB,early,F,68,False,232,63,156,0.288,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL15,li2024,MF,MF,,skin_unspecified,unspecified,IA,early,M,90,False,103,103,,,False,CD4_malignant
Li2024_atlas__CTCL16,li2024,MF,MF,,skin_unspecified,unspecified,IA,early,F,37,False,1814,1037,294,0.779,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL17,li2024,MF,MF,,skin_unspecified,unspecified,IIB,advanced,M,76,False,321,203,82,0.712,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL18,li2024,MF,MF,,skin_unspecified,unspecified,IV,advanced,M,68,False,7545,7004,127,0.982,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL2,li2024,MF,MF,,mixed,lesion,IA,early,M,67,False,5355,1357,2200,0.382,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL3,li2024,MF,MF,,mixed,lesion,IB,early,F,74,False,13308,8823,1778,0.832,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL4,li2024,MF,MF,,mixed,lesion,IB,early,M,57,False,12894,1241,7906,0.136,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL5,li2024,MF,MF,,mixed,lesion,IIB,advanced,M,56,False,14809,7532,4073,0.649,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL6,li2024,MF,MF,,mixed,lesion,IA,early,M,67,False,17933,3937,10298,0.277,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL7,li2024,MF,MF,,mixed,lesion,IA,early,F,59,False,56311,3658,38432,0.0869,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL8,li2024,MF,MF,,mixed,lesion,IB,early,M,61,False,12191,7379,2447,0.751,True,CD4_malignant|CD4_reactive
Li2024_atlas__CTCL9,li2024,MF,MF,,skin_unspecified,unspecified,IA,early,F,46,False,2585,2216,106,0.954,True,CD4_malignant|CD4_reactive
Li2024_atlas__MF15,li2024,MF,MF,CD4,plaque,lesion,IB,early,M,35,False,1317,,1147,,False,CD4_reactive
Li2024_atlas__MF17,li2024,MF,MF,CD4,tumor,lesion,IIB,advanced,M,36,False,5175,37,4385,0.00837,True,CD4_malignant|CD4_reactive
Li2024_atlas__MF18,li2024,MF,MF,CD4,tumor,lesion,IVA2,advanced,M,70,False,836,,600,,False,CD4_reactive
Li2024_atlas__MF21,li2024,MF,MF,CD4,tumor,lesion,IIB,advanced,M,79,False,3002,1761,708,0.713,True,CD4_malignant|CD4_reactive
Li2024_atlas__MF22,li2024,MF,MF,CD4,plaque,lesion,IB,early,M,71,False,1172,300,575,0.343,True,CD4_malignant|CD4_reactive
Li2024_atlas__MF26,li2024,MF,MF,CD4,plaque,lesion,IIA,early,M,15,False,2084,,1999,,False,CD4_reactive
Li2024_atlas__MF27,li2024,MF,MF,CD4,tumor,lesion,IIB,advanced,F,57,False,1746,107,725,0.129,True,CD4_malignant|CD4_reactive
Li2024_atlas__MF28,li2024,MF,MF,CD4,tumor,lesion,IIIB,advanced,M,60,False,15030,10395,1904,0.845,True,CD4_malignant|CD4_reactive
Li2024_atlas__MF30,li2024,MF,MF,CD4,tumor,lesion,IIB,advanced,F,48,False,5017,2342,1923,0.549,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT11,li2024,MF,MF,,patch,lesion,IVA2,advanced,M,50,False,2420,1948,282,0.874,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT35,li2024,MF,MF,,mixed,lesion,IIB,advanced,M,61,False,5080,3502,308,0.919,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT47,li2024,MF,MF,,mixed,lesion,IIB,advanced,M,60,False,571,307,165,0.65,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT50,li2024,MF,MF,,mixed,lesion,IB,early,M,73,False,3833,1464,1858,0.441,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT52,li2024,MF,MF,,mixed,lesion,IIB,advanced,F,55,False,266,160,89,0.643,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT53,li2024,MF,MF,,mixed,lesion,IIB,advanced,M,58,False,2555,1417,735,0.658,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT55,li2024,MF,MF,,mixed,lesion,IIB,advanced,M,71,False,2470,2177,71,0.968,True,CD4_malignant|CD4_reactive
Li2024_atlas__PT56,li2024,MF,MF,,mixed,lesion,IIB,advanced,M,65,False,5592,1292,2478,0.343,True,CD4_malignant|CD4_reactive
D7__MF309,rindler2021_mc,MF,MF,CD4,patch,lesion,IVA1,advanced,M,76,False,8778,5395,1748,0.755,True,CD4_malignant|CD4_reactive
D7__MF311,rindler2021_mc,MF,MF,CD4,patch,lesion,IVA1,advanced,F,74,False,9494,5693,1919,0.748,True,CD4_malignant|CD4_reactive
D7__P73,rindler2021_mc,MF,MF,,plaque,lesion,IIB,advanced,M,82,False,769,189,389,0.327,True,CD4_malignant|CD4_reactive
D7__P84,rindler2021_mc,MF,MF,CD4,plaque,lesion,IIB,advanced,M,75,False,6959,1892,3889,0.327,True,CD4_malignant|CD4_reactive
D3__P192,brunner2024,SS,SS,CD4,erythroderma,lesion,IVA1,advanced,M,67,False,4227,2874,660,0.813,True,CD4_malignant|CD4_reactive
D3__P87,brunner2024,SS,SS,CD4,erythroderma,lesion,IVA1,advanced,M,86,False,351,351,,,False,CD4_malignant
B5__PT09,buus2025,SS,SS,CD4,erythroderma,lesion,,,F,71,False,18455,10427,5273,0.664,True,CD4_malignant|CD4_reactive
B5__PT22,buus2025,SS,SS,CD4,erythroderma,lesion,,,M,70,False,9892,,7637,,False,CD4_reactive
B5__PT23,buus2025,SS,SS,CD4,erythroderma,lesion,,,F,85,False,8229,6371,1247,0.836,True,CD4_malignant|CD4_reactive
D6__MF2,gaydosik2019,SS,SS,CD4,erythroderma,lesion,IVA,advanced,F,68,False,1600,677,708,0.489,True,CD4_malignant|CD4_reactive
H__SS1,herrera2021,SS,SS,CD4,patch,lesion,IVB,advanced,F,49,False,137,36,73,0.33,True,CD4_malignant|CD4_reactive
H__SS3,herrera2021,SS,SS,CD4,plaque,lesion,IVA1,advanced,F,71,False,85,,52,,False,CD4_reactive
H__SS4,herrera2021,SS,SS,CD4,erythroderma,lesion,IVA2,advanced,M,56,False,288,204,46,0.816,True,CD4_malignant|CD4_reactive
D15__MF31,il4ra2026,SS,SS,,erythroderma,lesion,IVA1,advanced,M,89,True,50,,,,False,
D13__MF19,gaydosik2022,MF/SS,unresolved,,tumor,lesion,IIIB,advanced,M,62,False,528,253,235,0.518,True,CD4_malignant|CD4_reactive
D13__MF25,gaydosik2022,MF/SS,unresolved,CD4,erythroderma,lesion,IVA1,advanced,M,68,False,430,99,181,0.354,True,CD4_malignant|CD4_reactive
```

## §1e Arms

**What:** donors, cells and per-cell QC for every arm of the skin T/NK object; only the three CD4 arms (used_here = True) are analysed below.

**How:** (arm, donor) groups with >= 20 cells, independent cohort; median_* = median over the arm's donors of each donor's per-cell median.

```csv
arm,donors,cells,studies,median_cells_per_donor,median_umi_per_cell,median_genes_per_cell,median_doublet_score,used_here
CD4_malignant,78,187748,12,1.2e+03,3.33e+03,1.6e+03,0.0205,True
CD4_reactive,82,147233,12,620,2.04e+03,1.07e+03,0.0138,True
CD8,86,45562,13,197,2.11e+03,1.05e+03,0.0146,False
CD4_Treg,77,35560,13,215,2.55e+03,1.33e+03,0.0153,False
CD4_HC,21,4724,5,179,2.37e+03,870,0.0183,True
CD4_unassessed,2,71,1,35.5,1.27e+04,3.65e+03,0.0311,False
```

## §2a Gene universe

**What:** how many genes of each kind the screen tests.

**How:** surface = a-priori panel UNION confident surfaceome, intersected with the genes in the tables; lineage/trafficking = reference genes the builder added; a surface gene that is also a lineage or trafficking gene counts as surface.

```csv
gene_set,source,n_genes
lineage_control,-,28
other,-,2
surface,both,125
surface,panel,17
surface,surfaceome,1973
trafficking_control,-,210
```

40 genes form 42's contamination null (is_lineage_control) and are excluded from the shortlist.

## §2b Control genes present

**What:** whether each calibration gene can be tested.

**How:** positive controls are approved or registrational CTCL antigens; loss markers are the textbook CTCL loss phenotype. CD26 is an alias of DPP4, so its absence is expected.

```csv
gene,role,in_tables,gene_set
CCR4,positive_control,True,surface
TNFRSF8,positive_control,True,surface
KIR3DL2,positive_control,True,surface
IL2RA,positive_control,True,surface
CD7,loss_marker,True,surface
DPP4,loss_marker,True,surface
CD5,loss_marker,True,surface
CD6,loss_marker,True,surface
CD27,loss_marker,True,surface
CD2,loss_marker,True,surface
CD26,loss_marker,False,
```

## §3a Counts

**What:** how many surface genes are selectively up or down on malignant vs same-patient reactive CD4.

**How:** paired Wilcoxon on mean_all, BH over every tested gene (surface + controls).

**Full table on disk:** `tables/tgt49_all_genes.csv`

```csv
key,value
genes_tested_all_sets,2355
surface_genes_tested,2115
surface_q<0.05_up,388
surface_q<0.05_down,213
surface_q<0.05_up_and_log2fc>0.5,33
surface_q<0.05_down_and_log2fc<-0.5,62
median_paired_donors_per_gene,75.0
```

## §3b Surface genes UP on malignant CD4 (q < 0.05)

**What:** every surface gene whose expression is higher on malignant than on same-patient reactive CD4, sorted by effect.

**How:** paired Wilcoxon on mean_all, BH; no effect-size or prevalence gate (see §5 for that).

**Full table on disk:** `tables/tgt49_all_genes.csv`

```csv
gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm
CD70,CD70,1,75,1.1,0.92,3.09e-10,1.22,4.71e-11,0.21,0.0498,0.024,0.808,1.91,1.05e-07,True,1.0
CD9,,2,75,1.08,0.733,7.09e-07,1.12,1.55e-08,0.25,0.0583,0.107,0.718,0.771,0.0172,True,4.0
CPM,,3,75,1.02,0.747,7.08e-08,0.954,6.44e-09,0.137,0.0248,0.00669,0.577,1.89,4.34e-06,True,0.0
MRC2,,4,75,0.892,0.88,2.51e-09,0.896,1.78e-10,0.0805,0.0206,0.0087,0.449,1.28,8.37e-07,True,1.0
SLC29A1,,5,75,0.841,0.933,1.61e-10,0.899,4.71e-11,0.0825,0.0198,0.0125,0.397,0.987,1.05e-07,True,11.0
COL6A3,,6,75,0.811,0.88,3.28e-09,0.843,1.5e-10,0.113,0.0247,0.0158,0.564,1.47,2.98e-06,True,
CD151,,7,75,0.794,0.76,9.39e-08,0.946,1.77e-10,0.196,0.0889,0.0612,0.795,1.04,4.47e-05,True,4.0
ADGRE1,,8,75,0.792,0.707,7.67e-07,0.704,3.75e-08,0.0487,0.00961,0,0.308,1.15,5.69e-07,True,7.0
GRN,,9,75,0.778,0.907,9.81e-10,0.835,6.51e-11,0.211,0.103,0.041,0.846,1.58,2.29e-07,True,
NRN1,,10,75,0.776,0.773,1.35e-08,0.902,1.76e-09,0.0461,0.00362,0.00472,0.282,0.793,0.00053,True,0.0
TFRC,,11,75,0.731,0.893,1.67e-09,0.806,4.71e-11,0.296,0.147,0.118,0.949,0.806,4.47e-05,True,
SLC39A6,,12,75,0.707,0.947,1.91e-10,0.897,4.71e-11,0.183,0.0789,0.0447,0.91,1.2,2.11e-07,True,6.0
SLC4A2,,13,75,0.698,0.933,1.91e-10,0.739,4.71e-11,0.13,0.0406,0.0156,0.641,1.51,1.05e-07,True,
TNFRSF25,,14,75,0.698,0.747,9.14e-05,0.821,1.79e-07,0.339,0.171,0.0943,0.872,1.31,2.13e-06,True,1.0
SLC1A4,,15,75,0.69,0.747,5.23e-05,0.682,1.64e-07,0.144,0.0615,0.0239,0.628,1.38,2.59e-06,True,9.0
LGALS3,,16,75,0.663,0.587,0.00115,0.631,3.82e-05,0.427,0.292,0.361,0.885,0.21,0.184,True,
CD82,,17,75,0.632,0.773,1.41e-07,0.714,2.06e-09,0.477,0.283,0.176,1,1.04,1.32e-05,True,4.0
EVC2,,18,75,0.624,0.853,1.09e-08,0.657,1.57e-10,0.0536,0.0132,0,0.231,1.1,2.11e-07,True,1.0
SLC5A3,,19,75,0.617,0.893,1.81e-09,0.719,4.71e-11,0.217,0.0912,0.0499,0.769,1.45,7.22e-07,True,12.0
SGCB,,20,75,0.615,0.867,2.3e-08,0.727,2.05e-10,0.0735,0.0285,0.0204,0.372,0.76,2.42e-06,True,1.0
IL6R,,21,75,0.614,0.72,2.31e-05,0.778,1.51e-08,0.233,0.107,0.0726,0.769,0.857,0.000122,True,1.0
SYPL1,,22,75,0.599,0.907,1.15e-09,0.807,5.78e-11,0.362,0.2,0.179,0.949,0.674,0.000435,True,4.0
CALU,,23,75,0.587,0.88,6.18e-10,0.801,4.71e-11,0.19,0.0944,0.0392,0.846,1.26,5.31e-07,True,
QSOX1,,24,75,0.584,0.853,7.51e-09,0.714,1.06e-10,0.133,0.0677,0.0261,0.692,1.23,2.32e-07,True,1.0
VDAC1,,25,75,0.584,0.947,1.91e-10,0.645,4.71e-11,0.482,0.293,0.175,0.987,1.28,5.15e-07,True,
TSPAN3,,26,75,0.574,0.84,3.6e-09,0.717,1.1e-10,0.0966,0.0338,0.00943,0.462,1.28,5.69e-07,True,4.0
SLC39A14,,27,75,0.566,0.853,6.76e-09,0.717,2.59e-10,0.057,0.0178,0.0114,0.282,0.66,0.000261,True,6.0
TNFRSF4,,28,75,0.542,0.667,0.00308,0.538,1.81e-05,0.349,0.223,0.137,0.897,1.32,0.000118,True,1.0
TSPAN2,,29,75,0.53,0.747,1.24e-05,0.726,2.12e-08,0.135,0.0534,0.0377,0.654,0.841,0.000873,True,4.0
PTH2R,,30,75,0.53,0.667,2.53e-07,0.423,3.37e-08,0.0304,0.0017,0,0.308,0.795,1.02e-05,True,7.0
QSOX2,,31,75,0.512,0.853,7.08e-08,0.656,1.43e-10,0.0619,0.0182,0.00472,0.397,1.24,5.31e-07,True,1.0
GPI,,32,75,0.507,0.827,2.79e-07,0.618,7.24e-10,0.474,0.303,0.125,0.974,1.33,2.55e-05,True,
TMEM30A,,33,75,0.506,0.787,1.08e-08,0.686,4.71e-11,0.188,0.104,0.0612,0.846,1.04,9.01e-06,True,2.0
APP,,34,75,0.495,0.587,0.0017,0.669,4.66e-05,0.14,0.0725,0.0568,0.564,0.7,0.0114,True,1.0
MBP,,35,75,0.493,0.773,9.18e-08,0.561,2.71e-10,0.616,0.444,0.295,1,0.999,2.24e-05,True,
NEO1,,36,75,0.492,0.8,8.51e-07,0.435,1.69e-08,0.036,0.0133,0.0112,0.205,0.63,0.000188,True,1.0
FAM174B,,37,75,0.491,0.747,4.29e-07,0.546,8.07e-09,0.0409,0.00906,0.00415,0.269,0.767,0.00057,True,1.0
TMED9,,38,75,0.482,0.813,8.8e-09,0.642,7.36e-11,0.397,0.261,0.143,1,1.15,4.07e-07,True,
SIRPG,,39,75,0.48,0.64,0.000614,0.565,3.88e-06,0.239,0.132,0.025,0.744,1.93,1.32e-05,True,1.0
CAPNS1,,40,75,0.479,0.813,2.36e-07,0.649,8.68e-10,0.194,0.118,0.0396,0.872,1.38,1.81e-06,True,
LRPAP1,,41,75,0.478,0.893,4.95e-09,0.673,6.86e-11,0.356,0.219,0.0984,0.987,1.31,2.61e-07,True,
IL9R,,42,75,0.472,0.64,0.00147,0.541,3.9e-05,0.0693,0.0342,0.0261,0.41,0.377,0.0224,True,1.0
CD84,CD84 / SLAMF5,43,75,0.471,0.733,3.11e-05,0.61,3.7e-08,0.222,0.111,0.0469,0.795,1.28,1.31e-05,True,1.0
SLC7A1,,44,75,0.467,0.84,3.26e-07,0.653,3.4e-10,0.103,0.045,0.0141,0.5,1.32,7.44e-06,True,14.0
SLC52A2,,45,75,0.466,0.867,3.68e-09,0.682,8.77e-11,0.173,0.0997,0.0568,0.821,0.954,2.67e-06,True,10.0
RALA,,46,75,0.457,0.893,1.94e-09,0.677,6.69e-11,0.305,0.177,0.082,0.949,1.35,5e-07,True,
SDC4,,47,75,0.453,0.747,6.09e-06,0.594,4.22e-09,0.0859,0.0413,0.0341,0.423,0.672,0.000319,True,
SLC12A2,,48,75,0.449,0.84,3.93e-07,0.518,3.08e-10,0.0787,0.0322,0.0203,0.321,0.763,0.00203,True,12.0
NRROS,,49,75,0.438,0.667,9.14e-05,0.698,1.74e-07,0.101,0.0344,0.00341,0.5,1.52,5.22e-06,True,1.0
SLAMF1,,50,75,0.432,0.653,0.000238,0.64,1.32e-07,0.323,0.185,0.188,0.872,0.383,0.0176,True,1.0
C1QBP,,51,75,0.43,0.8,1.34e-07,0.509,3.23e-10,0.353,0.235,0.125,0.936,1.1,0.000102,True,
CHRNA6,,52,75,0.429,0.68,2.23e-06,0.451,8.26e-08,0.0315,0.00639,0,0.231,0.753,9.01e-06,True,4.0
ANXA2,,53,75,0.428,0.733,2.15e-05,0.482,4.91e-08,0.618,0.445,0.429,1,0.484,0.00256,True,
CLPTM1,,54,75,0.427,0.84,1.92e-08,0.632,6.86e-11,0.191,0.111,0.0469,0.833,1.05,2.35e-07,True,
HSPD1,,55,75,0.422,0.827,3.34e-07,0.427,2.05e-10,0.606,0.381,0.288,0.987,1.08,0.000143,True,
ENTPD1,,56,75,0.42,0.6,0.00714,0.496,0.000354,0.137,0.0585,0.0113,0.564,1.76,8.06e-05,True,2.0
NPTN,,57,75,0.419,0.893,1.74e-09,0.629,5.24e-11,0.152,0.0899,0.025,0.654,1.37,2.24e-06,True,1.0
CASD1,,58,75,0.414,0.84,2.41e-07,0.566,1.36e-09,0.074,0.0303,0,0.449,1.35,6.96e-07,True,14.0
P2RY11,,59,75,0.405,0.76,6.8e-06,0.457,1.39e-08,0.0531,0.0145,0.0087,0.269,0.74,1.83e-05,True,7.0
CD109,,60,75,0.403,0.68,0.000685,0.548,3.23e-06,0.0586,0.0283,0.0223,0.359,0.566,0.00561,True,0.0
FGL2,,61,75,0.4,0.627,0.00163,0.438,2.24e-05,0.0861,0.0429,0.0227,0.487,0.695,0.00362,True,
TMED4,,62,75,0.4,0.867,7.38e-09,0.61,4.71e-11,0.363,0.201,0.109,0.974,1.09,2.28e-07,True,
PRSS21,,63,75,0.399,0.587,3.44e-05,0.335,6.87e-06,0.0209,0.00526,0,0.41,0.448,0.000394,True,0.0
LPAR5,,64,75,0.399,0.707,3.59e-05,0.483,2.55e-07,0.0557,0.0196,0,0.218,1.14,5.62e-06,True,7.0
ATP1B3,,65,75,0.398,0.667,0.0163,0.474,7.24e-07,0.323,0.212,0.176,0.936,0.652,0.151,True,1.0
ENO1,,66,75,0.397,0.867,9.61e-09,0.358,6.68e-10,0.715,0.582,0.389,1,0.938,4.22e-07,True,
SLC19A1,,67,75,0.393,0.773,1.39e-06,0.496,3.64e-10,0.0517,0.0164,0,0.218,0.921,2.18e-07,True,12.0
BSG,CD147,68,75,0.392,0.747,2.21e-06,0.545,1.25e-09,0.432,0.298,0.205,1,0.6,5e-06,True,1.0
DMXL2,,69,75,0.391,0.84,2.71e-08,0.452,1.06e-09,0.0278,0.00261,0,0.115,0.61,2.18e-07,True,
GPR55,,70,75,0.389,0.653,0.000197,0.485,1.23e-06,0.0589,0.019,0.0156,0.346,0.64,0.00914,True,7.0
PDCD1,PD-1,71,75,0.387,0.573,0.0107,0.521,0.000334,0.147,0.087,0.0392,0.628,1.31,0.00536,True,1.0
TMEM87B,,72,75,0.385,0.733,1.71e-05,0.59,1.13e-09,0.165,0.0907,0.0301,0.756,1.38,3.06e-06,True,7.0
MCOLN1,,73,75,0.383,0.84,6.65e-09,0.457,8.48e-11,0.0562,0.0278,0.0102,0.218,0.787,2.24e-06,True,6.0
RAB5C,,74,75,0.379,0.853,3.6e-09,0.574,5.66e-11,0.365,0.256,0.159,1,0.906,3.37e-06,True,
PLXNA3,,75,75,0.377,0.813,1.53e-06,0.5,1.1e-09,0.0572,0.0271,0.0174,0.308,0.48,0.0212,True,1.0
ABCC1,,76,75,0.374,0.667,0.00132,0.577,6.73e-08,0.19,0.102,0.0425,0.705,1.39,9.79e-05,True,17.0
FZD6,,77,75,0.371,0.733,2.52e-05,0.45,1.46e-07,0.0424,0.0181,0,0.154,0.967,2.92e-05,True,7.0
IL10RB,,78,75,0.371,0.76,5.5e-06,0.645,5.81e-10,0.166,0.0907,0.0357,0.756,1.17,1.05e-07,True,1.0
TTYH3,,79,75,0.367,0.72,6.28e-07,0.509,4.92e-10,0.0341,0.013,0,0.244,0.84,1.27e-06,True,5.0
SLC16A1,,80,75,0.362,0.773,2.38e-06,0.589,4.48e-10,0.123,0.0692,0.0455,0.654,0.625,5.29e-05,True,12.0
GPR107,,81,75,0.361,0.787,1.07e-06,0.561,3.23e-10,0.113,0.0465,0.0249,0.526,0.953,6.05e-07,True,7.0
GNPTG,,82,75,0.359,0.773,3.75e-07,0.568,8.63e-11,0.206,0.124,0.0588,0.808,0.908,1.21e-06,True,
HNRNPU,,83,75,0.356,0.907,9.71e-10,0.402,4.71e-11,0.617,0.463,0.338,1,0.726,1.81e-06,True,
LAMP2,,84,75,0.355,0.893,3.15e-09,0.601,4.71e-11,0.174,0.0915,0.0503,0.808,0.925,8.9e-07,True,1.0
USP14,,85,75,0.354,0.92,1.69e-09,0.665,5.24e-11,0.266,0.157,0.0908,0.962,0.923,1.26e-06,True,
CANX,,86,75,0.353,0.853,3.28e-09,0.523,6.86e-11,0.429,0.295,0.14,1,1.36,9.15e-06,True,
LRFN4,,87,75,0.351,0.853,7.25e-09,0.411,1.73e-10,0.0297,0.00864,0,0.244,0.692,9.15e-06,True,1.0
SLC1A5,,88,75,0.35,0.667,0.000733,0.575,2.55e-07,0.214,0.131,0.0795,0.897,0.761,0.0014,True,10.0
SLC41A1,,89,75,0.348,0.853,2.73e-07,0.407,9.11e-11,0.0745,0.0331,0.0205,0.333,0.626,0.000204,True,9.0
CLDN15,,90,75,0.345,0.8,4.72e-07,0.375,9.15e-09,0.0276,0.00828,0,0.128,0.684,5.15e-07,True,4.0
ALCAM,,91,75,0.343,0.627,0.00124,0.374,1.65e-05,0.0517,0.0205,0.0164,0.359,0.557,0.00673,True,1.0
BMPR1A,,92,75,0.343,0.747,2.67e-05,0.368,6.97e-09,0.034,0.0199,0,0.141,0.911,0.000506,True,1.0
TNFRSF18,,93,75,0.343,0.627,0.0138,0.442,6.09e-05,0.369,0.25,0.235,0.833,0.339,0.0597,True,1.0
TM7SF3,,94,75,0.339,0.827,2.32e-08,0.514,8.77e-11,0.114,0.0554,0.0357,0.564,0.815,0.000104,True,7.0
ACVR2A,,95,75,0.338,0.747,5.48e-05,0.354,4.25e-07,0.0499,0.024,0.00559,0.295,1,0.000738,True,1.0
TM9SF1,,96,75,0.334,0.773,6.34e-07,0.578,7.24e-10,0.103,0.0604,0.0294,0.526,0.851,3.06e-06,True,9.0
TIMP1,,97,75,0.333,0.56,0.0114,0.516,3.9e-05,0.369,0.258,0.25,0.923,0.149,0.13,True,
ENOX2,,98,75,0.332,0.867,1.05e-07,0.497,2.37e-10,0.0634,0.0318,0.00943,0.218,0.975,5.26e-07,True,
TNFRSF8,CD30,99,75,0.332,0.667,0.000215,0.455,4.36e-07,0.038,0.0111,0,0.192,0.86,1.05e-07,True,1.0
BST2,,100,75,0.329,0.627,0.00416,0.409,1.03e-05,0.443,0.305,0.163,0.91,1.21,6e-05,True,1.0
LMAN2,,102,75,0.323,0.853,8.64e-09,0.548,7.25e-11,0.457,0.314,0.13,1,1.36,2.11e-07,True,1.0
TMX3,,103,75,0.322,0.773,1.5e-06,0.56,1.76e-10,0.142,0.0756,0.0204,0.705,1.39,3.81e-07,True,1.0
ANO6,,104,75,0.319,0.787,5.63e-06,0.556,4.2e-10,0.153,0.0863,0.0444,0.769,0.936,3.12e-05,True,8.0
ACP2,,105,75,0.319,0.773,1.68e-07,0.494,2e-10,0.0628,0.0281,0.00334,0.282,1.13,2.11e-07,True,1.0
PIGN,,106,75,0.319,0.88,1.41e-07,0.451,6.57e-10,0.0671,0.0327,0.0114,0.333,0.868,2.11e-07,True,
PIEZO1,,107,75,0.319,0.76,5.23e-05,0.528,3.64e-09,0.131,0.0812,0.0357,0.615,0.732,9.79e-06,True,39.0
IFNAR1,,108,75,0.319,0.747,6.92e-07,0.609,1.1e-10,0.184,0.115,0.0435,0.885,1.28,2.67e-07,True,1.0
WDR13,,109,75,0.316,0.8,3.53e-06,0.522,3.49e-10,0.0942,0.0512,0.0205,0.5,0.983,4.22e-07,True,
SLC12A9,,110,75,0.316,0.8,6.64e-07,0.523,3.94e-10,0.0765,0.0409,0.0124,0.372,1.01,8.9e-07,True,12.0
CHRNB1,,111,75,0.314,0.76,1.29e-05,0.424,7.3e-09,0.0561,0.0286,0.0142,0.167,0.689,1.89e-05,True,4.0
IL27RA,,112,75,0.313,0.68,0.000623,0.479,1.83e-08,0.304,0.191,0.148,0.897,0.42,0.00322,True,1.0
SCARB1,,113,75,0.311,0.76,1.38e-06,0.311,1.13e-08,0.0213,0.00433,0.00218,0.0769,0.46,0.000558,True,2.0
PHB2,,114,75,0.311,0.8,1.82e-07,0.424,1.3e-10,0.478,0.325,0.203,0.987,0.709,4.8e-05,True,
NOTCH1,,115,75,0.311,0.8,1.07e-06,0.425,1.44e-09,0.105,0.0616,0.0223,0.526,1.07,4.79e-06,True,1.0
NTRK2,,116,75,0.309,0.6,0.00112,0.437,1.16e-05,0.0494,0.0188,0.0279,0.321,0.0308,0.0977,True,1.0
CRELD2,,117,75,0.307,0.827,1.19e-07,0.532,2.1e-10,0.213,0.123,0.0795,0.808,0.75,8.67e-05,True,
B3GNT2,,118,75,0.304,0.693,0.000122,0.45,2.63e-09,0.23,0.134,0.0909,0.782,0.59,0.00147,False,
SLC43A3,,119,75,0.299,0.707,4.45e-05,0.441,2.17e-07,0.103,0.0441,0.0239,0.526,0.898,0.000143,True,12.0
DAGLB,,120,75,0.298,0.787,5.87e-06,0.406,3.24e-09,0.0467,0.022,0.00683,0.115,0.7,2.6e-05,True,4.0
RPN1,,121,75,0.297,0.787,1.18e-06,0.484,3.64e-10,0.376,0.266,0.141,0.974,1.03,6.95e-05,True,1.0
FLNA,,122,75,0.297,0.733,2.06e-05,0.367,6.97e-09,0.606,0.46,0.321,1,0.855,5e-06,True,
RPN2,,123,75,0.294,0.787,8.31e-07,0.497,2.05e-10,0.352,0.267,0.125,0.949,0.937,6.47e-06,True,
IFNGR2,,124,75,0.294,0.6,0.0117,0.46,8.94e-05,0.0707,0.0403,0.0348,0.372,0.307,0.0683,False,1.0
AAMP,,125,75,0.294,0.813,3.52e-08,0.559,6.69e-11,0.292,0.168,0.0781,0.923,1.21,9.5e-07,True,
L1CAM,,126,75,0.293,0.613,0.00154,0.386,7.89e-05,0.0357,0.0113,0.0171,0.192,0.122,0.371,True,1.0
YBX1,,127,75,0.293,0.893,1.56e-08,0.249,5.81e-10,0.746,0.648,0.58,1,0.237,0.0113,True,
FAM174A,,128,75,0.29,0.72,0.000163,0.487,9.91e-09,0.0615,0.0382,0.0192,0.269,0.661,0.00144,True,1.0
TNFSF4,,129,75,0.289,0.68,0.000592,0.295,6.72e-06,0.0416,0.0153,0.0102,0.231,0.542,0.00327,True,1.0
TMEM179B,,130,75,0.289,0.787,4.83e-07,0.541,2.71e-10,0.297,0.194,0.0767,0.936,1.5,5.15e-07,True,3.0
IL4R,,131,75,0.286,0.653,0.00136,0.563,1.18e-07,0.222,0.156,0.0781,0.885,1.07,1.02e-05,True,1.0
RNF13,,132,75,0.284,0.72,2.67e-05,0.474,5.94e-10,0.197,0.122,0.0669,0.885,0.885,3.42e-05,True,1.0
ATP6AP1,,133,75,0.284,0.707,3.5e-05,0.563,1.22e-09,0.234,0.156,0.0696,0.885,1.1,9.01e-06,True,
ATRAID,,134,75,0.283,0.773,2.34e-06,0.554,2.71e-10,0.312,0.218,0.109,0.974,0.824,8.42e-06,True,1.0
SCAP,,135,75,0.281,0.773,4.01e-06,0.434,5.03e-10,0.117,0.0611,0.0279,0.551,0.974,1.21e-06,True,8.0
TSPAN5,,136,75,0.28,0.68,0.000173,0.456,1.55e-08,0.132,0.0893,0.05,0.654,0.798,7.39e-05,True,4.0
SLC36A4,,137,75,0.279,0.813,3.03e-06,0.513,1.62e-09,0.075,0.0461,0.0217,0.359,0.82,0.000173,True,10.0
SLC5A6,,138,75,0.279,0.813,4.91e-07,0.375,1.08e-10,0.0527,0.0263,0.00472,0.205,0.798,4.25e-07,True,13.0
APLP2,,139,75,0.276,0.68,9.92e-05,0.561,1.1e-09,0.19,0.116,0.075,0.859,0.473,0.00125,True,1.0
C11orf24,,140,75,0.275,0.773,5.75e-07,0.541,4.48e-10,0.073,0.0313,0.0124,0.269,0.775,1.67e-06,True,1.0
CCDC126,,141,75,0.274,0.733,0.000166,0.344,1.02e-08,0.0526,0.0322,0.0179,0.218,0.528,2.41e-05,True,
SLC3A2,,142,75,0.273,0.773,1.92e-06,0.396,7.41e-10,0.466,0.353,0.279,0.987,0.439,0.00605,True,1.0
ICAM3,,143,75,0.269,0.813,5.43e-06,0.39,4.89e-09,0.568,0.44,0.325,0.987,0.537,0.000244,True,1.0
TM2D1,,144,75,0.268,0.72,2.85e-05,0.461,3.23e-10,0.151,0.0987,0.0435,0.808,1.1,1.21e-06,True,
HSP90AB1,,145,75,0.268,0.813,2.14e-08,0.169,4.48e-10,0.858,0.76,0.776,0.987,0.186,0.0767,True,
BMPR2,,146,75,0.268,0.693,0.000521,0.374,3.08e-08,0.0715,0.0453,0.0261,0.397,0.577,0.00382,True,1.0
LMBRD2,,147,75,0.267,0.653,0.00027,0.377,3.05e-09,0.0697,0.0401,0.0114,0.256,0.908,1.55e-06,True,9.0
SLC6A6,,148,75,0.264,0.773,1.16e-05,0.527,7.24e-10,0.181,0.0947,0.0588,0.744,0.921,2.55e-05,True,12.0
TRAC,,149,75,0.262,0.707,0.000335,0.419,7.92e-08,0.497,0.28,0.183,0.974,0.985,0.00418,False,
PIGT,,150,75,0.262,0.76,1.38e-05,0.486,1.14e-10,0.192,0.132,0.0755,0.872,0.848,1.52e-06,True,1.0
SLC39A8,,152,75,0.259,0.693,0.000457,0.486,1.24e-08,0.182,0.114,0.0742,0.769,0.728,0.000435,True,6.0
NCSTN,,153,75,0.259,0.72,6.8e-05,0.441,2.06e-09,0.105,0.0693,0.0196,0.577,0.953,2.35e-07,True,1.0
SGCE,,154,75,0.259,0.627,5.87e-06,0.273,3.08e-07,0.0294,0.00264,0,0.321,0.578,0.000104,True,1.0
PSEN1,,155,75,0.258,0.787,6.41e-06,0.482,3.64e-10,0.148,0.089,0.0408,0.641,0.751,2.3e-05,True,8.0
TMEM106B,,156,75,0.256,0.667,0.000634,0.483,1.52e-09,0.16,0.0917,0.0334,0.795,1.23,9.79e-06,True,1.0
SPN,,157,75,0.256,0.733,0.000246,0.471,6.54e-08,0.338,0.235,0.102,0.949,1.11,5.32e-07,True,1.0
LTBP3,,158,75,0.255,0.693,0.00132,0.611,6.25e-08,0.157,0.0878,0.0622,0.782,0.764,2.02e-05,True,
FLT1,,159,75,0.253,0.667,0.000456,0.194,8.4e-05,0.0151,0.00543,0,0.167,0.51,7.72e-05,True,1.0
TM2D3,,160,75,0.253,0.68,0.00128,0.521,1.2e-09,0.236,0.154,0.0984,0.949,0.66,3.22e-05,True,
P2RX7,,161,75,0.247,0.653,0.00163,0.266,1.29e-06,0.0301,0.0181,0.00679,0.115,0.566,0.00362,True,2.0
ICAM1,,162,75,0.247,0.613,0.0353,0.498,0.000105,0.111,0.0584,0.0542,0.526,0.282,0.0291,True,1.0
LMAN2L,,163,75,0.247,0.813,6.64e-07,0.323,3.6e-10,0.0269,0.0127,0,0.0897,0.681,5e-06,True,1.0
C12orf49,,164,75,0.247,0.747,1.07e-05,0.376,3.42e-09,0.0637,0.0468,0.0227,0.256,0.545,0.000786,True,
MERTK,,165,75,0.246,0.747,2.52e-07,0.201,6.54e-08,0.00907,0.00149,0,0.0641,0.292,9.87e-07,True,1.0
MFGE8,,167,75,0.245,0.707,0.000256,0.356,7.32e-08,0.0559,0.0295,0.0137,0.333,0.723,3.65e-05,True,
SEL1L,,168,75,0.243,0.733,0.000108,0.498,1.71e-10,0.178,0.109,0.0522,0.846,0.939,4.98e-06,True,
LRP10,,169,75,0.24,0.707,9.14e-05,0.476,2.33e-09,0.276,0.212,0.122,0.936,0.85,2.82e-06,True,1.0
OPN3,,171,75,0.237,0.787,3.76e-06,0.291,7.63e-09,0.0277,0.0125,0,0.141,0.632,5.78e-05,True,7.0
CPD,,172,75,0.235,0.627,0.0201,0.435,1.3e-05,0.13,0.103,0.0682,0.615,0.5,0.0105,True,1.0
ITGA3,,173,75,0.234,0.68,0.000274,0.294,1.88e-07,0.0262,0.0118,0,0.141,0.613,5.12e-06,True,1.0
ADAM17,,174,75,0.232,0.787,4.43e-06,0.423,2.05e-10,0.108,0.0674,0.0307,0.551,0.698,7.16e-05,True,1.0
TMEM87A,,175,75,0.232,0.773,2.62e-05,0.456,1.77e-10,0.209,0.137,0.0806,0.949,0.776,2.11e-07,True,7.0
TMEM123,,176,75,0.231,0.653,0.00236,0.386,2.66e-08,0.459,0.34,0.261,1,0.372,0.00954,True,1.0
INTS12,,177,75,0.229,0.8,1.74e-06,0.47,1.57e-10,0.0963,0.0498,0.0174,0.449,0.913,1.05e-07,True,
TGFBR3,,178,75,0.228,0.627,0.00747,0.495,1.06e-05,0.175,0.108,0.102,0.667,0.296,0.0667,True,1.0
SLC38A5,,179,75,0.227,0.6,0.034,0.355,0.000711,0.0581,0.0381,0.0102,0.359,0.738,0.000598,True,11.0
DLL3,,180,75,0.227,0.693,6.78e-07,0.262,9.37e-08,0.0134,0.0023,0,0.0769,0.296,0.0052,True,1.0
FAM189B,,181,75,0.227,0.747,9.29e-07,0.3,9.16e-11,0.0326,0.0153,0.00341,0.115,0.572,3.36e-06,True,4.0
SEMA4A,,182,75,0.227,0.6,0.0304,0.283,0.000966,0.0497,0.036,0.0112,0.282,0.734,0.00463,True,1.0
SLC9A7,,183,75,0.226,0.64,0.000698,0.359,1.32e-06,0.0368,0.013,0,0.192,0.76,2.07e-05,True,13.0
IGSF8,,184,75,0.226,0.653,0.000274,0.479,1.22e-09,0.153,0.0921,0.0401,0.705,0.976,4.79e-06,True,1.0
HM13,,185,75,0.222,0.733,0.00012,0.503,1.1e-09,0.369,0.254,0.125,0.962,0.859,5.31e-07,True,9.0
SSR1,,186,75,0.219,0.747,0.000685,0.431,2.26e-09,0.271,0.184,0.115,0.91,0.554,0.00023,True,1.0
ZDHHC5,,187,75,0.217,0.747,1.47e-05,0.396,5.03e-10,0.0888,0.0556,0.0307,0.385,0.539,3.53e-05,True,5.0
HEG1,,188,75,0.217,0.68,0.000623,0.365,5.86e-07,0.0899,0.0446,0.0227,0.449,0.77,0.000236,True,1.0
SLC31A1,,189,75,0.217,0.747,1.47e-05,0.329,1.11e-08,0.0377,0.0194,0.0087,0.128,0.588,0.000738,True,3.0
TXNDC15,,190,75,0.216,0.707,0.000157,0.507,7.15e-09,0.211,0.131,0.0656,0.859,0.791,0.00021,True,1.0
AEBP1,,191,75,0.216,0.64,7.51e-05,0.257,9.42e-06,0.0212,0.00633,0.00128,0.218,0.513,0.0155,True,
CLCN6,,192,75,0.215,0.813,1.02e-07,0.323,6.86e-11,0.0294,0.0125,0,0.0641,0.758,8.21e-07,True,
EPHB6,,193,75,0.215,0.627,0.00349,0.474,7.49e-08,0.0815,0.0455,0.0328,0.449,0.18,0.0127,True,1.0
SLC30A1,,194,75,0.214,0.747,0.000634,0.275,5.36e-09,0.0624,0.0315,0.025,0.346,0.226,0.0105,True,6.0
TLL1,,195,75,0.214,0.707,2.71e-08,0.204,2.21e-09,0.0098,0,0,0.0897,0.309,1.83e-05,True,
IGSF3,,196,75,0.213,0.813,6.04e-08,0.199,5.45e-09,0.00993,0.00165,0,0.0641,0.297,9.01e-06,True,1.0
ENO2,,197,75,0.211,0.653,0.0228,0.449,5.77e-06,0.144,0.0942,0.0478,0.615,0.553,0.0243,True,
IGHM,,198,75,0.211,0.68,0.000337,0.279,2.77e-06,0.0271,0.00575,0,0.269,0.633,3.59e-06,True,
CD99L2,,199,75,0.211,0.72,2.73e-05,0.39,8.42e-09,0.056,0.0292,0.00472,0.205,0.974,2.52e-06,True,
MPZL1,,200,75,0.209,0.653,0.000592,0.328,2.9e-08,0.0361,0.0165,0.00472,0.0897,0.635,8.77e-05,True,1.0
ITFG1,,201,75,0.209,0.72,6.65e-05,0.414,5.68e-10,0.161,0.0971,0.0392,0.705,0.991,3.42e-06,True,1.0
PKD1,,203,75,0.208,0.733,9.76e-05,0.276,7.3e-09,0.0522,0.0308,0.0112,0.256,0.699,0.000111,True,11.0
ATP13A3,,204,75,0.204,0.733,0.000964,0.48,8.83e-10,0.117,0.0663,0.0196,0.551,0.951,1.48e-06,True,11.0
ADCY3,,205,75,0.204,0.707,0.000382,0.413,5.62e-08,0.0487,0.0286,0.0109,0.282,0.654,3.85e-06,True,12.0
ATP6V0A2,,206,75,0.203,0.72,0.00128,0.403,7.52e-09,0.109,0.0647,0.0283,0.551,0.904,5.61e-05,True,8.0
IL21R,,207,75,0.203,0.627,0.00695,0.534,3.98e-06,0.239,0.154,0.075,0.795,1.29,1.62e-05,True,1.0
NCLN,,208,75,0.201,0.667,0.000891,0.459,2.94e-09,0.084,0.0528,0.0134,0.449,1.16,6.88e-07,True,
KIAA1324L,,209,75,0.201,0.64,0.00093,0.279,5.18e-06,0.0328,0.00825,0.00472,0.192,0.541,0.0177,True,1.0
CD46,,211,75,0.199,0.72,0.00021,0.461,8.14e-10,0.292,0.216,0.137,0.962,0.645,0.00125,True,1.0
CHRNA5,,212,75,0.198,0.76,8.76e-07,0.322,1.33e-09,0.0209,0.00448,0.00341,0.0641,0.326,0.00165,True,4.0
SLC26A6,,213,75,0.198,0.653,0.00105,0.295,2.28e-07,0.0318,0.0183,0.00669,0.154,0.485,3.74e-05,True,8.0
CREG1,,214,75,0.197,0.72,0.00185,0.344,1.35e-07,0.0683,0.0418,0.0205,0.295,0.55,0.00095,True,
ADAM9,,215,75,0.197,0.72,9.56e-05,0.272,9.39e-09,0.0238,0.0137,0.00639,0.0769,0.286,0.000761,True,1.0
NOTCH2,,216,75,0.196,0.68,0.00326,0.465,7.71e-09,0.1,0.0602,0.025,0.5,0.871,1.2e-06,True,1.0
RGMB,,217,75,0.195,0.64,0.00102,0.335,2.59e-08,0.0485,0.0307,0.0204,0.269,0.0606,0.0977,True,0.0
SLC41A3,,218,75,0.194,0.613,0.00509,0.399,5.36e-08,0.091,0.0632,0.0312,0.449,0.686,7.94e-05,True,10.0
GALNT11,,219,75,0.194,0.68,0.00706,0.423,5.76e-08,0.115,0.0697,0.0281,0.538,0.852,2.42e-05,True,
MFSD2A,,220,75,0.19,0.747,7.9e-05,0.277,1.1e-07,0.0231,0.0086,0.0087,0.0256,0.124,0.0384,True,11.0
ITGB8,,221,75,0.188,0.627,0.000138,0.19,1.45e-05,0.0126,0.00348,0,0.115,0.335,0.000578,True,1.0
GPR158,,222,75,0.187,0.68,6.04e-08,0.216,1.44e-08,0.00739,0,0,0.0897,0.195,9.01e-06,True,7.0
EXTL2,,223,75,0.186,0.72,1.02e-05,0.218,4.39e-08,0.0167,0.00511,0,0.0128,0.437,1.57e-05,True,
PTPRK,,224,75,0.185,0.707,7.86e-05,0.18,1.25e-06,0.0199,0.00739,0,0.154,0.504,0.000737,True,1.0
LDLR,,225,75,0.184,0.68,0.0177,0.356,1.25e-05,0.0982,0.0617,0.0455,0.5,0.513,0.00327,True,1.0
BTLA,,227,75,0.182,0.533,0.0384,0.176,0.00445,0.0401,0.0285,0.0189,0.269,0.393,0.0473,True,1.0
STT3B,,228,75,0.182,0.667,0.0135,0.364,1.69e-08,0.276,0.189,0.0705,0.846,1.39,5.47e-05,True,11.0
CALR,,229,75,0.181,0.68,0.000382,0.222,4.3e-08,0.593,0.514,0.426,1,0.462,0.00486,True,
SPPL2A,,230,75,0.18,0.693,0.000733,0.502,3.24e-09,0.226,0.159,0.1,0.872,0.477,0.00023,True,9.0
FUT4,,231,75,0.179,0.747,2.56e-05,0.183,8.83e-09,0.017,0.00788,0,0.115,0.435,5.56e-07,False,
DAG1,,232,75,0.178,0.747,1.12e-05,0.28,3.86e-10,0.0395,0.0246,0,0.167,0.935,0.000122,True,1.0
GINM1,,233,75,0.178,0.693,0.000592,0.42,1e-09,0.138,0.0929,0.058,0.731,0.594,4.76e-06,True,1.0
IL2RG,,235,75,0.176,0.667,0.00574,0.17,3.5e-06,0.72,0.699,0.375,1,0.855,6.92e-06,True,1.0
CHST2,,236,75,0.176,0.693,0.00035,0.31,7.79e-07,0.0426,0.0154,0,0.179,0.898,4.04e-05,False,
PTPRA,,237,75,0.174,0.667,0.0311,0.401,3.42e-09,0.208,0.14,0.0656,0.744,0.952,0.000282,True,1.0
MSN,,238,75,0.173,0.667,0.00378,0.335,5.51e-08,0.597,0.489,0.25,1,1.18,0.000143,True,
CD3D,,239,75,0.172,0.653,0.00104,0.283,9.14e-08,0.755,0.602,0.471,1,0.806,1.42e-05,True,1.0
C1orf159,,240,75,0.171,0.68,0.000583,0.318,5.91e-08,0.0357,0.0219,0.0125,0.0385,0.479,0.000173,True,1.0
PLXNB2,,241,75,0.169,0.72,0.000268,0.187,3.37e-06,0.0147,0.00471,0,0.103,0.403,2.34e-07,True,1.0
TMX4,,242,75,0.168,0.72,0.000311,0.46,1.13e-09,0.299,0.19,0.163,0.936,0.413,0.011,True,1.0
SLC38A9,,243,75,0.166,0.72,0.000999,0.375,6.97e-09,0.0587,0.0344,0.00669,0.179,1.06,3.08e-07,True,11.0
TM9SF3,,244,75,0.165,0.613,0.0148,0.414,1.31e-08,0.181,0.12,0.0682,0.769,0.685,0.000217,True,8.0
CD164,,245,75,0.165,0.72,0.0158,0.37,5.62e-08,0.401,0.3,0.197,0.974,0.653,0.00136,True,1.0
CELSR3,,246,75,0.165,0.76,5.07e-06,0.214,1.23e-07,0.014,0.00346,0,0.103,0.333,2.95e-06,True,7.0
TMEM25,,247,75,0.162,0.613,0.0213,0.218,1.14e-05,0.0266,0.0158,0,0.0513,0.692,0.00188,True,1.0
SIDT2,,248,75,0.161,0.693,0.00128,0.289,5.36e-08,0.0348,0.023,0.0112,0.0513,0.372,0.00014,True,10.0
STIM1,,249,75,0.161,0.68,0.000315,0.451,5.18e-10,0.167,0.105,0.046,0.705,0.944,1.37e-05,True,1.0
C6orf120,,250,75,0.16,0.68,0.0026,0.374,2.38e-08,0.0601,0.0343,0.00341,0.295,1.05,1.49e-05,True,
SLC22A4,,253,75,0.157,0.72,3.93e-07,0.145,1.23e-08,0.00778,0.00103,0,0.0513,0.237,0.000181,True,12.0
ADCY9,,254,75,0.155,0.573,0.00449,0.164,1.71e-05,0.0112,0.0048,0,0.103,0.367,0.000154,True,12.0
PANX1,,255,75,0.155,0.64,0.0022,0.287,1.56e-07,0.0397,0.0259,0.0156,0.167,0.492,0.00147,True,4.0
HSP90AA1,,257,75,0.153,0.747,0.000233,0.112,1.48e-09,0.879,0.81,0.836,1,0.197,0.0946,True,
TMEM245,,258,75,0.153,0.653,0.00993,0.423,1.11e-08,0.15,0.11,0.0492,0.718,0.875,4.8e-05,True,14.0
PLXNA1,,259,75,0.153,0.64,0.000725,0.162,7.79e-07,0.0134,0.00538,0,0.0641,0.349,1.17e-05,True,1.0
IGFLR1,,260,75,0.153,0.573,0.0431,0.368,2.05e-05,0.303,0.252,0.148,0.885,0.583,0.00393,True,1.0
DPP7,,261,75,0.152,0.693,0.0019,0.418,6.44e-09,0.415,0.331,0.191,0.962,0.996,2.89e-05,True,
PLAT,,262,75,0.151,0.667,0.000129,0.188,9.4e-06,0.0113,0.00327,0,0.0769,0.326,0.000373,True,
NTRK1,,263,75,0.15,0.667,0.000272,0.197,6.72e-06,0.0157,0.00687,0.00218,0.103,0.298,0.0187,True,1.0
ACVR2B,,264,75,0.148,0.68,0.00554,0.167,1.23e-06,0.0149,0.00789,0,0.0641,0.47,3.47e-05,True,1.0
RYK,,265,75,0.147,0.68,0.00429,0.304,1.49e-07,0.0538,0.0325,0.0236,0.256,0.435,0.00954,True,1.0
TMEM104,,266,75,0.146,0.68,0.000723,0.233,7.32e-08,0.0319,0.0181,0,0.103,0.765,7.79e-07,True,11.0
LTBP1,,267,75,0.146,0.72,2.97e-05,0.147,1.92e-07,0.00874,0.00164,0,0.0641,0.232,0.00029,True,
CCR10,,269,75,0.143,0.573,0.0336,0.126,0.00157,0.0297,0.0226,0.0087,0.269,0.496,0.0465,True,7.0
SEPTIN2,,270,75,0.142,0.613,0.000964,0.352,1.15e-07,0.261,0.211,0.159,0.808,0.622,0.0131,True,
ATP1A1,,271,75,0.141,0.627,0.0278,0.353,7.15e-09,0.346,0.297,0.245,0.962,0.0451,0.34,True,10.0
CTNS,,273,75,0.139,0.613,0.00714,0.202,4.84e-06,0.0265,0.0152,0,0.0897,0.618,5e-07,True,7.0
HYAL2,,274,75,0.139,0.8,3.8e-07,0.232,7.21e-10,0.0217,0.00644,0,0.0385,0.477,0.000232,True,0.0
ST3GAL4,,275,75,0.139,0.693,0.00601,0.234,2.35e-06,0.0296,0.0172,0.0142,0.128,0.246,0.162,True,
SLC12A5,,276,75,0.138,0.627,6.38e-05,0.149,4.56e-06,0.00855,0.00421,0.0087,0.128,-0.0896,0.401,True,12.0
HRH1,,277,75,0.138,0.733,1.46e-05,0.166,5.95e-07,0.0101,0.00168,0,0.0641,0.296,2.98e-06,True,7.0
TMCO3,,278,75,0.138,0.64,0.00674,0.365,5.15e-07,0.0696,0.0399,0.0223,0.308,0.589,0.000578,True,10.0
NRP1,,280,75,0.136,0.64,0.000225,0.127,1.3e-05,0.00846,0.00192,0,0.0513,0.277,0.00224,True,1.0
ADAM10,,282,75,0.133,0.64,0.00556,0.419,1.33e-09,0.25,0.149,0.0612,0.859,1.04,5.25e-06,True,1.0
STX4,,283,75,0.126,0.64,0.0472,0.32,1.39e-08,0.162,0.119,0.0719,0.833,0.587,0.000385,True,
SLC45A4,,284,75,0.125,0.64,0.00422,0.179,3.08e-07,0.0153,0.0112,0,0.167,0.479,0.000493,True,12.0
MEGF8,,285,75,0.124,0.693,0.000561,0.194,1.51e-07,0.0152,0.00726,0,0.0513,0.396,2.35e-07,True,1.0
IL1RAP,,288,75,0.121,0.613,0.0175,0.232,2.57e-05,0.0328,0.0253,0.0205,0.269,0.162,0.0986,True,1.0
CLIC4,,289,75,0.119,0.653,0.00348,0.224,1.12e-05,0.0221,0.012,0.0102,0.141,0.215,0.232,True,
ABCC5,,290,75,0.119,0.613,0.015,0.28,1.42e-07,0.04,0.0225,0.00472,0.103,0.721,6.9e-06,True,12.0
ATRN,,291,75,0.118,0.653,0.00182,0.283,2.12e-08,0.0455,0.0264,0.0112,0.282,0.627,2.9e-05,True,1.0
MFAP3,,292,75,0.118,0.68,0.0113,0.222,3e-07,0.0284,0.0181,0,0.0256,0.737,5.45e-05,True,1.0
FKRP,,293,75,0.118,0.707,0.000157,0.204,4.39e-08,0.0214,0.0104,0,0.115,0.505,5.32e-07,True,1.0
LRRC8A,,296,75,0.116,0.667,0.00098,0.304,8.83e-10,0.058,0.0398,0.0125,0.321,0.594,3.01e-05,True,
LRP5,,297,75,0.115,0.68,1.32e-06,0.135,1.17e-08,0.00651,0.00117,0,0.0385,0.192,9.06e-05,True,1.0
TSPAN13,,298,75,0.114,0.627,0.00142,0.147,6.29e-06,0.0119,0.00567,0,0.128,0.345,0.000115,True,4.0
SLC22A5,,299,75,0.114,0.613,0.0173,0.171,4.21e-07,0.0225,0.0135,0.00436,0.0385,0.329,0.00249,True,12.0
GPR19,,301,75,0.113,0.693,0.00427,0.172,5.51e-07,0.0126,0.00551,0,0.0641,0.309,8.9e-07,True,7.0
SLC17A5,,303,75,0.112,0.613,0.0441,0.344,1.64e-07,0.0741,0.0472,0.0189,0.256,0.859,0.000143,True,12.0
TMEM63B,,304,75,0.112,0.667,0.00503,0.259,1.73e-08,0.0374,0.0224,0.0087,0.115,0.634,5.8e-06,True,10.0
CD276,,306,75,0.11,0.64,0.000159,0.145,1.03e-05,0.00989,0.00251,0,0.0897,0.268,0.00416,True,1.0
SEMA4B,,308,75,0.109,0.68,0.0377,0.183,3.64e-05,0.0194,0.0125,0,0.0256,0.551,5.2e-06,True,1.0
EFNA4,,310,75,0.108,0.573,0.0213,0.185,6.54e-08,0.0333,0.0207,0.0174,0.0769,0.16,0.295,True,0.0
LRP6,,311,75,0.108,0.587,0.000218,0.0981,3.27e-05,0.00834,0.00208,0,0.0641,0.277,0.000187,True,1.0
MMGT1,,312,75,0.107,0.627,0.0331,0.416,2.12e-08,0.113,0.0747,0.0341,0.603,0.615,0.00618,True,
DCBLD1,,313,75,0.105,0.6,0.0205,0.183,5.75e-05,0.0163,0.00826,0,0.0256,0.399,0.000878,True,1.0
DGCR2,,314,75,0.105,0.627,0.0269,0.348,1.67e-09,0.0981,0.0693,0.0312,0.474,0.732,5e-06,True,1.0
STS,,315,75,0.104,0.733,9.41e-06,0.12,2.78e-07,0.00891,0.00324,0,0.0128,0.25,0.000365,True,2.0
FZD1,,318,75,0.103,0.653,1.18e-07,0.0969,2.3e-08,0.00444,0,0,0.0513,0.174,6.11e-06,True,7.0
CLDN1,,322,75,0.102,0.653,0.00261,0.0923,0.000129,0.00902,0.00256,0.00384,0.141,0.131,0.552,True,4.0
LTB4R,,323,75,0.0981,0.653,0.0116,0.161,6.86e-07,0.0218,0.0109,0.0087,0.154,0.308,0.0134,True,7.0
FGFR2,,325,75,0.0969,0.6,1.16e-05,0.0799,1e-06,0.00462,0.00113,0,0.154,0.166,0.00486,True,1.0
ERMAP,,326,75,0.0959,0.653,0.00549,0.184,1.1e-06,0.0168,0.0111,0.00472,0.0256,0.318,0.002,True,1.0
EFNA3,,330,75,0.0931,0.613,0.0224,0.124,5.43e-05,0.0124,0.00719,0.00683,0.0256,0.0971,0.687,True,0.0
APLP1,,331,75,0.093,0.747,5.37e-07,0.0916,3.75e-08,0.00395,8.71e-05,0,0.0256,0.117,0.00549,True,1.0
PODXL2,,332,75,0.0927,0.653,0.000342,0.12,6.89e-06,0.00764,0.00308,0,0.0385,0.252,0.0241,True,1.0
GPC1,,333,75,0.0921,0.613,0.0212,0.107,0.000374,0.00932,0.00509,0.00128,0.0385,0.211,0.0354,True,0.0
LAMA3,,334,75,0.0919,0.667,6.04e-06,0.102,8.9e-08,0.00463,0.000364,0,0.0256,0.169,0.00442,True,
SLC23A2,,335,75,0.0901,0.693,0.00685,0.332,2.25e-08,0.0501,0.0327,0.00683,0.218,0.9,1.21e-06,True,12.0
HHLA2,,336,75,0.0896,0.653,2.25e-07,0.122,6.77e-08,0.0064,4.86e-05,0,0.167,0.184,0.00412,True,1.0
FZD5,,337,75,0.0888,0.72,2.76e-05,0.0911,4.73e-08,0.00517,0.000963,0,0,0.157,0.0394,True,7.0
SPPL2B,,340,75,0.0863,0.68,0.01,0.312,1.35e-08,0.0549,0.0335,0.00871,0.244,0.684,4.94e-05,True,9.0
HNRNPK,,341,75,0.0859,0.627,0.011,0.227,1.73e-08,0.626,0.541,0.443,1,0.307,0.00188,True,
VIPR2,,344,75,0.0815,0.56,0.000151,0.142,8.65e-06,0.00856,0.00189,0,0.128,0.3,0.00374,True,7.0
ADAM22,,345,75,0.0802,0.573,0.00105,0.157,1.18e-05,0.0118,0.00628,0,0.0256,0.304,0.00317,True,1.0
MUC1,,346,75,0.0802,0.613,0.000614,0.0827,9.98e-07,0.00778,0.00182,0,0.0769,0.203,5.69e-07,True,1.0
PCDHGA10,,350,75,0.0753,0.68,5.5e-06,0.0768,3.71e-07,0.0046,6.32e-05,0,0.0256,0.153,2.98e-06,True,1.0
PTGFRN,,352,75,0.0737,0.68,0.00135,0.127,1.04e-05,0.0102,0.0045,0,0.0256,0.263,0.000376,True,1.0
CDON,,356,75,0.0716,0.653,0.000187,0.0909,6.39e-06,0.00513,0.00164,0,0.0513,0.119,0.00023,True,1.0
PCDHGA6,,358,75,0.0703,0.667,1.26e-05,0.0641,2.36e-06,0.00306,0,0,0.0128,0.0972,1.48e-06,True,1.0
CLDN12,,360,75,0.068,0.733,1.36e-05,0.106,1.43e-07,0.00707,0.00146,0,0,0.199,1.07e-05,True,4.0
DSCAML1,,361,75,0.0673,0.707,2.18e-05,0.0966,9.44e-07,0.00719,0.00108,0,0.0256,0.18,0.0036,True,1.0
SCN8A,,362,75,0.0671,0.627,0.000405,0.0637,1.94e-06,0.00368,0.00137,0,0.0641,0.134,0.000114,True,24.0
LRFN1,,364,75,0.0661,0.6,0.0311,0.114,2.2e-05,0.0116,0.00515,0,0.0385,0.311,8.72e-05,True,1.0
CRLF2,,366,75,0.0645,0.587,0.004,0.0645,0.00053,0.00748,0.00302,0,0.103,0.278,3.22e-05,True,1.0
FZD3,,367,75,0.064,0.573,0.0417,0.138,0.00021,0.0134,0.00772,0,0.141,0.415,0.000593,True,7.0
IL17RC,,369,75,0.0634,0.64,0.0019,0.0899,4.26e-06,0.00616,0.00317,0,0,0.191,0.0169,True,1.0
IL5RA,,371,75,0.0622,0.587,0.0285,0.0692,0.00191,0.00651,0.00412,0,0.0385,0.189,0.0136,True,1.0
ULBP1,,372,75,0.0618,0.64,0.000398,0.0686,1.23e-06,0.00342,0.000616,0,0,0.119,8.78e-05,True,0.0
TSPAN33,,377,75,0.0575,0.6,0.0168,0.0783,0.000407,0.00905,0.00198,0,0.0513,0.245,0.000354,True,4.0
NLGN3,,378,75,0.0565,0.653,0.00092,0.0489,4.72e-06,0.00288,0.000654,0,0.0128,0.0735,0.000437,True,1.0
LRP11,,380,75,0.0561,0.533,0.0278,0.0895,0.000422,0.0102,0.00792,0.00384,0.115,0.142,0.0204,True,1.0
NPR2,,381,75,0.0555,0.68,0.00112,0.0686,4.4e-06,0.00487,0.00157,0,0,0.128,7.03e-05,True,1.0
SLC16A4,,382,75,0.055,0.72,2.25e-06,0.0911,2.73e-09,0.00448,0.000774,0,0,0.129,0.000493,True,12.0
NRP2,,385,75,0.0535,0.613,0.0298,0.0266,0.00147,0.00427,0.00175,0,0.0256,0.156,1.45e-05,True,1.0
THY1,,390,75,0.0513,0.587,0.000548,0.142,7.23e-07,0.0216,0.00279,0.00226,0.192,0.549,0.00338,True,0.0
MARCKS,,391,75,0.0512,0.613,0.0146,0.117,2.55e-05,0.0164,0.00821,0,0.115,0.467,0.00942,True,
S1PR2,,392,75,0.051,0.64,0.0426,0.153,2.75e-05,0.0198,0.0111,0.00472,0.0385,0.28,0.0163,True,7.0
PLXNB1,,394,75,0.0502,0.64,0.000647,0.0728,4.9e-06,0.00492,0.00194,0,0.0385,0.18,0.000938,True,1.0
TMEFF1,,395,75,0.0487,0.64,7.66e-06,0.0458,4.24e-07,0.00208,0,0,0.0128,0.0643,9.07e-05,True,1.0
ITGB4,,396,75,0.048,0.627,0.0222,0.045,5.53e-05,0.00564,0.00167,0,0,0.156,0.000166,True,1.0
DDR2,,398,75,0.0466,0.64,0.00441,0.0495,3.23e-05,0.00628,0.00259,0,0.0128,0.185,0.0221,True,1.0
SV2A,,400,75,0.045,0.52,0.0242,0.142,0.00169,0.0143,0.00849,0,0.0641,0.433,0.00283,True,12.0
UMODL1,,401,75,0.0439,0.613,5.07e-06,0.0462,5.69e-07,0.00187,0,0,0.0256,0.0641,0.00023,True,1.0
TYRO3,,404,75,0.0426,0.64,4.52e-06,0.0493,4.39e-08,0.00261,0.000318,0,0.0256,0.0862,0.000761,True,1.0
ITGA10,,406,75,0.0393,0.613,0.00959,0.0397,2.42e-05,0.00301,0.00139,0,0,0.101,3.3e-05,True,1.0
FN1,,408,75,0.0389,0.6,0.0188,0.053,4.5e-06,0.00982,0.0035,0.00384,0,0.106,0.167,True,
SLC6A8,,409,75,0.0386,0.587,0.00244,0.0346,5.59e-05,0.00383,0.00133,0,0,0.111,0.0693,True,12.0
MMP17,,410,75,0.0381,0.6,0.000201,0.0375,8.24e-05,0.00239,0,0,0.0641,0.0754,0.0008,True,0.0
FBN1,,412,75,0.0375,0.56,0.00755,0.0724,6.49e-05,0.00674,0.00127,0,0.0641,0.235,0.022,True,
TLR2,,413,75,0.0375,0.56,0.0086,0.0245,0.000391,0.0036,0.00131,0,0.0256,0.108,0.000833,True,1.0
ADRA2B,,414,75,0.0375,0.587,0.0091,0.0406,0.000378,0.0023,0.00106,0,0.0128,0.0723,0.0369,True,7.0
TMEM63C,,415,75,0.0373,0.587,0.002,0.0453,8e-05,0.00318,0.000645,0,0.0128,0.0981,2.3e-05,True,10.0
SCNN1D,,416,75,0.0368,0.587,0.00278,0.0419,8.9e-05,0.00288,0.00114,0,0,0.101,0.000206,True,2.0
FGFR1,,418,75,0.0363,0.52,0.043,0.15,0.00252,0.0452,0.0307,0.0205,0.346,0.289,0.164,True,1.0
ICAM5,,420,75,0.0351,0.613,3.88e-05,0.0436,2.17e-06,0.00266,0,0,0,0.0839,0.0489,True,1.0
SLC9A3,,423,75,0.0327,0.56,0.0062,0.108,2.74e-07,0.0117,0.00564,0,0.0128,0.296,0.000422,True,10.0
LRRC24,,425,75,0.0319,0.587,0.00189,0.0384,2.51e-05,0.00248,0.000744,0,0,0.0674,1.57e-05,True,1.0
SLC29A4,,427,75,0.0311,0.56,0.00618,0.0324,0.000475,0.00297,0.00157,0,0,0.114,0.0321,True,10.0
HTR2B,,429,75,0.0307,0.587,0.000482,0.044,1.24e-05,0.00173,0,0,0,0.054,0.000113,True,7.0
PCDH12,,433,75,0.0299,0.627,3.78e-06,0.032,1.56e-07,0.00159,0,0,0.0128,0.0405,0.000188,True,1.0
GPR63,,438,75,0.0284,0.587,0.000723,0.0275,2.18e-05,0.00307,0.000942,0,0,0.0893,0.00133,True,7.0
SORCS1,,439,75,0.0281,0.587,2.35e-06,0.0281,4.3e-07,0.0011,0,0,0.0641,0.0307,0.0657,True,1.0
MUC16,,441,75,0.0275,0.56,0.000293,0.0258,3.6e-05,0.00323,0.000542,0,0.103,0.103,0.00254,True,1.0
SLCO5A1,,445,75,0.0245,0.64,1.36e-05,0.0596,1.04e-06,0.00518,0.000518,0,0.0513,0.115,3.39e-05,True,12.0
EPHB4,,448,75,0.0236,0.653,0.0225,0.0309,0.000287,0.00285,0.000214,0,0.0128,0.0847,3.42e-05,True,1.0
PTCHD1,,449,75,0.0235,0.573,4.9e-07,0.0202,1.34e-07,0.000878,0,0,0.128,0.0263,3.09e-05,True,11.0
LYPD5,,451,75,0.0234,0.627,0.000508,0.0421,2.17e-06,0.00313,0,0,0,0.0735,0.00324,True,0.0
OTOA,,455,75,0.0216,0.6,3.17e-05,0.0228,9.66e-07,0.000969,0,0,0,0.0296,0.000217,True,0.0
ITGA7,,456,75,0.021,0.587,0.00486,0.0196,0.000348,0.00144,0,0,0,0.0381,0.000545,True,1.0
ACHE,,458,75,0.02,0.533,0.0228,0.0278,0.000875,0.00182,0.000521,0,0,0.0675,0.0208,True,0.0
DNER,,460,75,0.0194,0.56,0.000949,0.0275,0.000198,0.00137,0.000628,0,0.115,0.043,0.217,True,1.0
TAS1R3,,462,75,0.0174,0.56,0.000379,0.0156,7.11e-06,0.000544,0,0,0.0128,0.0227,0.000203,True,7.0
THSD1,,464,75,0.0165,0.547,0.021,0.0259,0.000489,0.00139,0.000489,0,0,0.047,0.000394,True,1.0
WNT5B,,467,75,0.0154,0.573,0.000262,0.022,8.71e-06,0.00129,0,0,0.0385,0.0545,0.00147,True,
OPRL1,,469,75,0.0148,0.52,0.0152,0.0324,0.000104,0.00183,0.000627,0,0,0.048,0.00262,True,7.0
GRIN1,,471,75,0.0144,0.547,7.37e-05,0.014,2.35e-06,0.000612,0,0,0,0.0198,0.000685,True,3.0
UPK1A,,472,75,0.0143,0.587,0.00239,0.0166,0.000355,0.00137,0,0,0,0.0374,0.000196,True,4.0
EPCAM,,480,75,0.0128,0.56,8.88e-05,0.0132,1.2e-05,0.000795,0,0,0.103,0.0214,0.0325,True,1.0
SDK2,,481,75,0.0126,0.547,0.0454,0.016,0.00836,0.00292,0.00157,0,0.0256,0.139,0.0489,True,1.0
ITGA2B,,487,75,0.0111,0.6,1.05e-05,0.00969,4.05e-07,0.000915,0,0,0,0.0242,0.000103,True,1.0
PODXL,,489,75,0.0109,0.547,0.0431,0.042,7.11e-05,0.00305,0.00171,0.00341,0.0128,-0.0245,0.871,True,1.0
AQP10,,490,75,0.0107,0.547,0.00588,0.0101,0.000434,0.00068,0,0,0,0.0221,0.00165,True,5.0
SUSD2,,491,75,0.00997,0.56,0.0117,0.0199,6.64e-05,0.00147,0,0,0.0128,0.0412,0.000323,True,1.0
AOC3,,495,75,0.00916,0.533,0.00586,0.0132,0.00029,0.000552,0,0,0,0.0202,0.145,True,1.0
FAM171A2,,496,75,0.00897,0.507,0.00133,0.00835,0.00017,0.000736,0,0,0,0.0223,0.000282,True,1.0
TSPAN1,,498,75,0.00863,0.573,0.000471,0.0135,6.09e-05,0.000677,0,0,0,0.0174,0.000155,True,4.0
PILRA,,500,75,0.00823,0.52,0.0269,0.0158,0.00117,0.00311,0.00122,0,0,0.0754,0.000139,True,1.0
FAT1,,505,75,0.00592,0.52,0.0181,0.016,0.00263,0.00155,0,0,0,0.0629,0.0492,True,1.0
TM4SF5,,506,75,0.00564,0.52,0.000883,0.00411,0.000173,0.000223,0,0,0,0.00646,0.000732,True,4.0
PTPRG,,508,75,0.00432,0.507,0.0137,0.00375,0.00833,0.000469,0,0,0,0.0156,0.00517,True,1.0
S1PR3,,511,75,0.00321,0.507,0.0228,0.0114,0.000247,0.000803,0,0,0,0.026,0.00393,True,7.0
EPDR1,,515,75,0.00254,0.52,0.0341,0.012,0.00456,0.00287,0.00162,0,0.0128,0.0867,0.000399,True,
BST1,,519,75,0.00134,0.507,0.0108,0.00506,0.00123,0.000494,0,0,0,0.0136,0.0415,True,0.0
BDKRB2,,521,75,0.000574,0.507,0.00116,0.00325,0.000426,5.67e-05,0,0,0.0385,0.00231,0.0147,True,7.0
```

## §3c Surface genes DOWN on malignant CD4 (q < 0.05)

**What:** every surface gene whose expression is lower on malignant than on same-patient reactive CD4, sorted by effect (most lost first). These are candidates for loss-of-marker diagnostics, not for targeting.

**How:** paired Wilcoxon on mean_all, BH.

**Full table on disk:** `tables/tgt49_all_genes.csv`

```csv
gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm
CD7,CD7,2115,75,-2.85,0.0933,3.6e-09,-2.28,2.59e-10,0.038,0.539,0.211,0.308,-2.06,0.0526,True,1.0
CD8A,,2114,75,-2.28,0.0933,3.84e-08,-1.46,1.05e-07,0.0107,0.148,0.082,0.115,-1.5,0.00633,True,1.0
KLRD1,,2113,75,-2.15,0.0267,1.61e-10,-1.26,6.86e-11,0.00301,0.103,0.041,0.0256,-1.32,0.00828,False,
PRF1,,2112,75,-1.79,0.227,0.00013,-1.14,0.00416,0.0612,0.194,0.0849,0.346,-0.757,0.472,True,
KLRK1,,2111,75,-1.75,0.2,8.67e-06,-1.16,0.00172,0.00946,0.139,0.0696,0.231,-1.41,0.207,True,1.0
ITGA4,,2110,75,-1.65,0.04,1.91e-10,-1.09,1.67e-10,0.0217,0.165,0.0392,0.141,-0.593,0.306,True,1.0
SELL,,2109,75,-1.63,0.347,0.0127,-0.905,0.139,0.149,0.257,0.0357,0.577,1.17,0.00821,True,1.0
KLRB1,,2108,75,-1.44,0.227,0.00565,-0.904,0.0905,0.0469,0.194,0.181,0.359,-2.01,0.0961,True,1.0
TYROBP,,2107,75,-1.3,0.12,2.82e-08,-0.675,5.76e-08,0.0169,0.0738,0.0568,0.0769,-1.27,0.0187,True,
CCR6,,2106,75,-1.26,0.28,0.00422,-0.826,0.0296,0.0254,0.114,0.152,0.321,-2.06,0.0291,True,7.0
CD8B,,2104,75,-1.24,0.04,9.12e-06,-0.763,2.63e-05,0,0.0452,0.0909,0.0513,-2.21,8.86e-05,True,1.0
TRDC,,2103,75,-1.22,0.0933,1.09e-08,-0.722,4.59e-08,0.000816,0.0344,0.0269,0.0513,-0.885,0.00166,False,
CD300A,,2102,75,-1.19,0.0267,2.37e-10,-0.8,6.86e-11,0.0015,0.0469,0.0339,0.0128,-0.977,0.000768,True,1.0
PTGER2,,2101,75,-1.14,0.08,7.25e-09,-0.737,2.85e-08,0.00172,0.0433,0.0239,0.0641,-0.84,0.00346,True,7.0
CXCR6,,2100,75,-1.09,0.28,0.0275,-0.739,0.0744,0.0255,0.121,0.0714,0.295,-0.924,0.451,True,7.0
KLRC1,,2099,75,-1.06,0.16,0.000734,-0.632,0.00155,0.00101,0.0436,0.0156,0.141,-0.629,0.128,True,1.0
CLEC2B,,2098,75,-1.04,0.2,3.92e-08,-0.572,0.0001,0.201,0.356,0.219,0.654,-0.639,0.26,True,
LAG3,,2097,75,-0.991,0.267,0.0013,-0.636,0.0341,0.0268,0.112,0.0341,0.282,-0.218,0.482,True,1.0
GPR132,,2096,75,-0.958,0.173,2.98e-08,-0.637,1.97e-06,0.0244,0.118,0.0447,0.269,-0.603,0.62,True,7.0
S1PR1,,2095,75,-0.934,0.32,0.00027,-0.596,0.0167,0.0613,0.109,0.0201,0.436,0.669,0.0115,True,7.0
KLRG1,,2094,75,-0.921,0.04,1.61e-10,-0.524,1.77e-10,0.0182,0.0606,0.0375,0.0256,-0.54,0.0629,True,1.0
TGFBR2,,2093,75,-0.897,0.187,2.25e-08,-0.471,6.79e-05,0.0965,0.197,0.118,0.487,-0.596,0.657,True,1.0
FCGR3A,,2092,75,-0.858,0.08,4.05e-09,-0.477,5.15e-09,0.00107,0.0216,0.0102,0.0256,-0.416,0.0369,True,1.0
PTGDR,,2091,75,-0.853,0.0267,4.26e-10,-0.574,9.16e-11,0.000514,0.0276,0.0142,0.0256,-0.636,0.00068,True,7.0
DPP4,,2090,75,-0.841,0.253,0.00391,-0.575,0.0257,0.0087,0.0457,0.0492,0.179,-1.15,0.267,True,1.0
KLRF1,,2089,75,-0.809,0.0267,4.14e-10,-0.478,1.73e-10,0.000123,0.0211,0.00895,0,-0.418,0.00221,True,1.0
ITGB7,,2088,75,-0.801,0.32,0.00503,-0.357,0.265,0.134,0.168,0.0957,0.551,-0.0351,0.447,True,1.0
IL10RA,,2087,75,-0.783,0.187,1.32e-08,-0.472,6.49e-05,0.206,0.32,0.218,0.782,-0.427,0.264,True,1.0
ANK3,,2086,75,-0.779,0.253,9.95e-05,-0.438,0.00457,0.0253,0.0622,0.0204,0.244,-0.0854,0.515,True,
IL7R,,2085,75,-0.775,0.107,1.15e-09,-0.392,7.35e-06,0.41,0.601,0.547,0.91,-0.559,0.0986,True,1.0
TMEM63A,,2084,75,-0.763,0.16,3.05e-08,-0.45,1.47e-06,0.0199,0.0599,0.0279,0.141,-0.243,0.817,True,10.0
LPAR6,,2083,75,-0.761,0.28,0.0136,-0.437,0.185,0.0555,0.141,0.275,0.385,-2.15,0.0467,True,7.0
DPEP2,,2082,75,-0.76,0.147,4.72e-09,-0.491,7.24e-09,0.00368,0.0265,0.00341,0,-0.0117,0.894,True,0.0
PECAM1,,2081,75,-0.731,0.173,3.03e-06,-0.461,8.15e-06,0.00958,0.0432,0.0234,0.0769,-0.454,0.392,True,1.0
AREG,,2080,75,-0.73,0.253,0.0041,-0.366,0.0774,0.0809,0.157,0.179,0.41,-1.33,0.104,True,1.0
CXCR4,,2079,75,-0.7,0.12,2.5e-09,-0.279,3.78e-06,0.542,0.741,0.672,0.987,-0.793,0.0271,True,7.0
SLAMF7,,2078,75,-0.699,0.16,7.12e-06,-0.396,2.63e-05,0.00506,0.0302,0,0.0897,0.179,0.234,True,1.0
FAS,,2077,75,-0.674,0.333,0.0272,-0.231,0.549,0.14,0.165,0.115,0.615,-0.168,0.47,True,1.0
ITGA6,,2076,75,-0.672,0.133,5.54e-09,-0.408,1.88e-07,0.0184,0.0525,0.0409,0.0513,-0.632,0.0378,True,1.0
P2RY8,,2075,75,-0.664,0.213,7.66e-06,-0.162,0.0106,0.0842,0.155,0.0625,0.41,0.164,0.378,True,7.0
SERINC5,,2074,75,-0.663,0.213,3.24e-05,-0.229,0.00473,0.0922,0.117,0.0816,0.423,-0.0685,0.634,True,9.0
VSIR,,2073,75,-0.662,0.107,1.44e-08,-0.41,8.6e-06,0.092,0.177,0.104,0.436,-0.176,0.616,True,1.0
CD55,,2072,75,-0.65,0.253,7.84e-06,-0.289,0.00988,0.196,0.281,0.27,0.705,-0.779,0.361,True,0.0
HAVCR2,,2070,75,-0.636,0.187,0.000286,-0.366,0.0296,0.028,0.0584,0.0203,0.192,0.0121,0.725,True,1.0
LGALS3BP,,2069,75,-0.629,0.2,1.12e-05,-0.357,5.52e-05,0.00609,0.0368,0.00226,0.0641,0.111,0.124,True,
FLT3LG,,2068,75,-0.618,0.187,3.53e-06,-0.21,0.04,0.185,0.284,0.277,0.859,-0.659,0.322,True,1.0
CSF1,,2067,75,-0.617,0.28,0.0117,-0.332,0.214,0.019,0.0444,0.0271,0.205,-0.188,0.558,True,1.0
KLRC3,,2066,75,-0.613,0.08,3.28e-09,-0.369,1.13e-08,0.000136,0.0216,0,0.0385,0.00419,1,True,1.0
TMIGD2,,2065,75,-0.613,0.133,8.19e-05,-0.395,0.000256,0.00089,0.0232,0.0135,0.0897,-0.527,0.181,True,1.0
RNF130,,2064,75,-0.601,0.28,0.00038,-0.292,0.0216,0.0199,0.0462,0.0279,0.128,-0.38,0.549,True,1.0
IL18RAP,,2063,75,-0.564,0.307,0.00136,-0.257,0.0841,0.00957,0.032,0.0114,0.128,-0.157,0.585,True,1.0
GPR183,,2062,75,-0.562,0.24,1.83e-05,-0.165,0.0539,0.238,0.259,0.224,0.744,-0.361,0.207,True,7.0
LAIR1,,2061,75,-0.551,0.173,2.15e-05,-0.349,0.000106,0.00506,0.0343,0.0164,0.103,-0.295,0.963,True,1.0
ITGA5,,2060,75,-0.54,0.24,6.53e-06,-0.219,0.00252,0.0281,0.061,0.0236,0.179,-0.0516,0.504,True,1.0
GCNT1,,2059,75,-0.536,0.267,0.00526,-0.277,0.156,0.0274,0.0404,0.0348,0.192,-0.143,0.893,False,
ICAM2,,2058,75,-0.536,0.347,0.000315,-0.0721,0.586,0.145,0.181,0.0625,0.692,0.584,0.00452,True,1.0
TRGC1,,2057,75,-0.529,0.2,0.000337,-0.304,0.00622,0.00656,0.0259,0.0189,0.154,-0.4,0.868,False,
VAMP5,,2056,75,-0.521,0.347,0.0103,-0.0283,0.744,0.0995,0.139,0.125,0.5,-0.454,0.881,True,
CD69,,2055,75,-0.518,0.187,1.92e-06,-0.126,0.00156,0.502,0.609,0.662,0.949,-0.805,0.063,True,1.0
CD244,,2054,75,-0.508,0.213,0.000475,-0.295,0.0073,0.00618,0.0319,0.0164,0.141,-0.393,0.96,True,1.0
ADRB2,,2053,75,-0.507,0.16,9.08e-07,-0.237,0.000861,0.0228,0.0415,0.0377,0.0897,-0.53,0.427,True,7.0
SLC12A7,,2052,75,-0.501,0.213,6.62e-06,-0.191,0.00867,0.0179,0.0566,0.0205,0.128,-0.309,0.375,True,12.0
FCRL3,,2050,75,-0.484,0.213,0.01,-0.315,0.017,0.00313,0.0218,0,0.0897,0.0995,0.0415,True,1.0
ITPRIP,,2049,75,-0.476,0.213,1.1e-05,-0.237,0.0341,0.0489,0.0915,0.0522,0.269,-0.341,1,True,
HCST,,2047,75,-0.472,0.333,0.000242,-0.126,0.218,0.498,0.554,0.45,0.949,-0.11,0.709,True,
ITGAM,,2046,75,-0.463,0.12,2.67e-06,-0.242,3.01e-05,0.00206,0.0168,0.00669,0.0897,-0.275,0.316,True,1.0
S1PR5,,2045,75,-0.46,0.0533,4.14e-10,-0.304,9.38e-11,0,0.0123,0.00559,0,-0.305,0.0251,True,7.0
NCAM1,,2044,75,-0.452,0.133,9e-07,-0.303,5.44e-07,0,0.0128,0.00256,0.0256,-0.104,0.195,True,1.0
IL12RB2,,2043,75,-0.449,0.173,4.45e-05,-0.212,0.00674,0.015,0.036,0,0.192,0.438,0.00322,True,1.0
BTN3A1,,2041,75,-0.431,0.173,2.12e-07,-0.123,0.0367,0.0955,0.113,0.05,0.487,0.401,0.0265,True,1.0
GPR18,,2040,75,-0.411,0.307,0.0019,-0.14,0.274,0.0243,0.0339,0.00415,0.103,0.412,0.00324,True,7.0
PLXDC1,,2039,75,-0.411,0.147,9.57e-07,-0.23,7.11e-06,0.00346,0.0166,0.024,0.0641,-0.76,0.0948,True,1.0
CCR5,,2037,75,-0.409,0.187,0.00042,-0.279,0.00225,0.00517,0.0235,0.00384,0.0769,0.0198,0.256,True,7.0
SLC12A6,,2036,75,-0.409,0.227,1.5e-06,-0.112,0.0231,0.0386,0.0514,0.0181,0.179,0.38,0.0164,True,12.0
BTN3A2,,2035,75,-0.405,0.253,1.26e-05,-0.0392,0.691,0.176,0.187,0.0469,0.808,0.789,0.00101,True,1.0
CD38,CD38,2034,75,-0.402,0.133,2.37e-08,-0.26,1.46e-07,0.0021,0.0134,0,0,0.0689,0.884,True,1.0
ABCB1,,2030,75,-0.38,0.12,3.38e-06,-0.202,6.42e-05,0.00138,0.0132,0,0.0641,0.047,0.776,True,12.0
ENG,,2028,75,-0.374,0.253,0.0211,-0.205,0.0348,0.00802,0.0236,0,0.103,0.254,0.0155,True,1.0
CD247,,2027,75,-0.374,0.213,0.00509,-0.0169,0.709,0.386,0.441,0.205,0.949,0.504,0.00172,False,
LRRC32,,2026,75,-0.373,0.307,0.0279,-0.178,0.129,0.00498,0.0152,0.00334,0.0385,0.0364,0.0977,True,1.0
ADA,,2025,75,-0.37,0.28,0.00541,-0.176,0.523,0.0321,0.0653,0.0205,0.282,0.103,0.352,True,
TGFBR1,,2024,75,-0.362,0.307,0.00757,-0.114,0.521,0.0413,0.0537,0.0196,0.179,0.226,0.205,True,1.0
CD48,,2023,75,-0.36,0.307,0.000206,-0.0914,0.882,0.465,0.474,0.413,0.962,-0.222,0.623,True,0.0
EPHA1,,2020,75,-0.343,0.16,5.98e-07,-0.149,7.19e-05,0.00521,0.0164,0,0.0256,0.137,0.139,True,1.0
BTN3A3,,2019,75,-0.335,0.2,1.16e-05,-0.0145,0.793,0.0727,0.0737,0.00683,0.321,1.06,1.55e-06,True,1.0
FCGRT,,2017,75,-0.332,0.347,0.0424,-0.115,0.599,0.0266,0.0425,0.0357,0.167,-0.171,0.884,True,1.0
SLC44A2,,2016,75,-0.33,0.36,0.000949,0.0128,0.869,0.111,0.103,0.0239,0.59,1.02,0.000261,True,10.0
IL18R1,,2014,75,-0.325,0.267,0.00262,-0.0793,0.516,0.052,0.0623,0.0196,0.256,0.425,0.0676,True,1.0
EVI2B,,2012,75,-0.321,0.267,4.65e-05,-0.0195,0.424,0.264,0.295,0.235,0.91,0.00857,0.903,True,1.0
KIT,,2011,75,-0.32,0.0933,3.36e-07,-0.175,4.77e-07,0.000786,0.00951,0.0142,0.0128,-0.53,0.0144,True,1.0
AMIGO2,,2010,75,-0.316,0.32,0.0417,-0.119,0.36,0.0102,0.0194,0.0083,0.128,-0.054,0.27,True,1.0
ITGAX,,2009,75,-0.301,0.16,6.9e-06,-0.143,8.64e-05,0.002,0.0101,0.00256,0.0641,-0.0148,0.877,True,1.0
CCR2,,2008,75,-0.298,0.213,0.00011,-0.175,0.000563,0.00205,0.0138,0,0.0385,0.0687,0.689,True,7.0
RELL1,,2007,75,-0.298,0.32,0.00161,0.0299,0.862,0.0617,0.0625,0.0375,0.308,0.339,0.0716,True,1.0
SELPLG,,2006,75,-0.293,0.32,0.00139,-0.0321,0.549,0.307,0.324,0.107,0.795,1.09,0.000703,True,1.0
FCRL6,,2005,75,-0.288,0.12,4.31e-09,-0.154,1.44e-08,0.000379,0.00943,0,0.0128,0.0113,0.922,True,1.0
IL1R1,,2004,75,-0.287,0.293,0.0151,-0.135,0.104,0.00916,0.0193,0.00451,0.0769,0.123,0.276,True,1.0
ITM2B,,2003,75,-0.285,0.227,6.09e-06,-0.0568,0.297,0.618,0.679,0.656,0.962,-0.562,0.21,True,1.0
S1PR4,,2002,75,-0.285,0.28,9.14e-05,0.0711,0.709,0.161,0.16,0.0455,0.705,1.07,0.000191,True,7.0
CD53,,2001,75,-0.281,0.267,5.58e-05,-0.000889,0.975,0.435,0.478,0.391,1,-0.0128,0.64,True,4.0
PGAP1,,2000,75,-0.281,0.32,0.000561,-0.0276,0.413,0.0398,0.0452,0.0261,0.218,0.0801,0.843,True,7.0
PMEPA1,,1999,75,-0.278,0.28,0.00105,-0.0915,0.0396,0.0072,0.0207,0.0087,0.0897,0.00934,0.61,True,1.0
GPR15,,1998,75,-0.277,0.293,0.0366,-0.132,0.178,0.00195,0.0148,0,0.128,0.075,0.342,True,7.0
PRNP,,1997,75,-0.272,0.32,0.021,0.0651,0.242,0.293,0.289,0.311,0.91,-0.493,0.217,True,0.0
LPAR2,,1995,75,-0.255,0.36,0.0278,0.00572,0.465,0.0687,0.0747,0.0236,0.308,0.561,0.00593,True,7.0
TNFSF13B,,1993,75,-0.246,0.253,0.000343,-0.0687,0.0489,0.0129,0.0186,0.00472,0.0128,0.24,0.02,True,1.0
CX3CR1,,1991,75,-0.246,0.227,0.000311,-0.112,0.00342,0.00226,0.0103,0,0.0128,0.0718,0.603,True,7.0
TSPAN18,,1990,75,-0.245,0.227,0.00024,-0.143,0.000943,0.00204,0.0094,0,0.0256,0.067,0.202,True,4.0
C3AR1,,1989,75,-0.235,0.16,1.07e-06,-0.121,2.42e-05,0.00219,0.00992,0.00218,0.0128,-0.00329,0.681,True,7.0
CD160,,1987,75,-0.232,0.24,3.73e-05,-0.144,0.00102,0.00148,0.0111,0.00639,0.0256,-0.203,0.274,True,0.0
VIPR1,,1986,75,-0.232,0.213,9.84e-06,-0.119,0.000124,0.00254,0.00698,0,0,0.0623,0.262,True,7.0
PTPRM,,1984,75,-0.229,0.28,0.00544,-0.117,0.121,0.00307,0.00989,0,0.0513,0.0952,0.248,True,1.0
GPR25,,1983,75,-0.227,0.0267,1.69e-09,-0.138,3.4e-10,0,0.00526,0,0,0,0.649,True,7.0
VNN2,,1982,75,-0.22,0.253,0.00129,-0.107,0.0529,0.00265,0.00888,0,0.0128,0.0679,0.00125,True,0.0
SIGIRR,,1981,75,-0.22,0.293,0.00134,0.108,0.178,0.254,0.251,0.133,0.897,0.337,0.00618,True,1.0
CASP7,,1979,75,-0.218,0.333,0.00278,0.0116,0.759,0.0336,0.0383,0.0124,0.115,0.307,0.000923,True,
PLXND1,,1978,75,-0.214,0.32,0.0112,-0.0681,0.215,0.0115,0.0189,0.0124,0.0897,-0.17,0.492,True,1.0
ITGA1,,1977,75,-0.214,0.253,0.0192,-0.149,0.0618,0.0033,0.0143,0,0.103,0.145,0.0012,True,1.0
CD101,,1976,75,-0.211,0.187,3.44e-05,-0.087,0.00691,0.00163,0.00944,0.0087,0.0385,-0.249,0.472,True,1.0
IFNGR1,,1975,75,-0.21,0.36,0.00904,0.0483,0.194,0.115,0.133,0.0896,0.628,0.135,0.526,True,1.0
VCAM1,,1974,75,-0.209,0.227,0.000256,-0.0911,0.003,0.00283,0.0103,0,0.0513,0.114,1.79e-05,True,1.0
IGFBP3,,1973,75,-0.205,0.147,3.54e-07,-0.0767,4.53e-05,0.00232,0.00908,0,0,0.0919,0.46,True,
MS4A1,,1972,75,-0.203,0.187,2.06e-05,-0.0794,0.000183,0.00187,0.0104,0.0134,0.0128,-0.457,0.0514,True,4.0
SLC14A1,,1971,75,-0.201,0.24,0.000165,-0.0748,0.00421,0.00153,0.00866,0,0,0.0472,0.0946,True,8.0
EMB,,1970,75,-0.2,0.373,0.0204,0.0835,0.154,0.348,0.317,0.218,0.936,0.276,0.146,True,1.0
PRSS23,,1969,75,-0.197,0.307,0.00142,-0.0726,0.0513,0.00641,0.0111,0.00683,0.0128,-0.0328,0.94,True,
CD52,CD52,1966,75,-0.19,0.373,0.0296,0.00704,0.683,0.752,0.781,0.676,0.987,0.0821,0.714,True,0.0
CXCL16,,1964,75,-0.188,0.333,0.00367,-0.0357,0.26,0.00856,0.0156,0.0174,0,-0.276,0.392,True,1.0
ABCA7,,1963,75,-0.188,0.253,2.03e-05,0.00785,0.658,0.0343,0.0364,0.0167,0.192,0.305,0.00376,True,15.0
CFH,,1958,75,-0.174,0.227,0.00145,-0.0681,0.00252,0.00358,0.0113,0.0125,0.0641,-0.337,0.128,True,
IGF2R,,1957,75,-0.172,0.387,0.022,0.142,0.0349,0.102,0.108,0.0269,0.526,0.935,0.000873,True,1.0
THBS1,,1955,75,-0.171,0.293,0.0311,-0.0608,0.21,0.0105,0.0157,0.0156,0.0641,-0.395,0.737,True,
SLC39A10,,1953,75,-0.168,0.307,0.0328,0.117,0.0179,0.0924,0.0878,0.0409,0.436,0.202,0.0217,True,7.0
LYPD3,,1952,75,-0.167,0.307,0.00113,0,0.819,0.0468,0.0389,0.0625,0.218,-0.513,0.108,True,0.0
SLC16A7,,1951,75,-0.166,0.32,0.00641,0.0584,0.827,0.0909,0.0836,0.0375,0.474,0.352,0.26,True,12.0
P2RY14,,1950,75,-0.166,0.2,0.000114,-0.0869,0.000769,0.00208,0.00839,0,0.0128,0.0572,0.00482,True,7.0
DLL1,,1949,75,-0.166,0.28,0.0269,-0.0793,0.239,0.00453,0.00858,0.0109,0.0256,-0.254,0.211,True,1.0
IQGAP2,,1948,75,-0.165,0.333,0.00601,0.0862,0.375,0.219,0.249,0.112,0.833,0.846,0.0127,True,
IL1R2,,1947,75,-0.162,0.28,0.0083,-0.0568,0.0776,0.00661,0.012,0,0.0385,0.218,0.00196,True,1.0
SLC8B1,,1946,75,-0.16,0.347,0.0119,0.0501,0.164,0.0443,0.041,0.0166,0.103,0.45,0.00759,True,13.0
CXCR5,,1943,75,-0.154,0.32,0.00701,-0.0543,0.047,0.00393,0.00861,0,0.0513,0.123,0.0049,True,7.0
RNF149,,1942,75,-0.154,0.4,0.00449,0.176,0.0165,0.378,0.366,0.204,0.974,0.591,0.00463,True,1.0
KRT1,,1940,75,-0.152,0.28,0.00439,-0.0028,0.737,0.0261,0.024,0.129,0.397,-1.68,0.238,True,
MPZL2,,1938,75,-0.15,0.227,0.000805,-0.0641,0.00223,0.00253,0.0109,0.00903,0.0256,-0.282,0.0679,True,1.0
IL17RA,,1937,75,-0.15,0.4,0.0328,0.199,0.00531,0.151,0.128,0.0408,0.679,0.944,0.000131,True,1.0
CD37,CD37,1935,75,-0.149,0.293,6.65e-05,0.0252,0.582,0.682,0.664,0.516,0.987,0.123,0.189,True,4.0
CYSLTR2,,1933,75,-0.148,0.32,0.0227,-0.0768,0.123,0.0039,0.00807,0.00654,0.0769,-0.138,0.935,True,7.0
CD2,,1931,75,-0.146,0.453,0.0478,0.0676,1,0.6,0.583,0.392,0.897,0.278,0.00618,True,1.0
SLC43A1,,1930,75,-0.145,0.32,0.0222,-0.034,1,0.0163,0.0187,0.00895,0.0256,0.112,0.0493,True,12.0
EVI2A,,1929,75,-0.144,0.387,0.00747,0.109,0.14,0.205,0.205,0.0844,0.949,0.78,0.00261,True,1.0
CALY,,1925,75,-0.138,0.12,6.28e-07,-0.0641,1.49e-05,0.000444,0.00445,0,0.0128,0.0123,0.502,True,1.0
LTB4R2,,1924,75,-0.135,0.293,0.0245,-0.0411,0.744,0.0104,0.0158,0,0.0128,0.331,0.12,True,7.0
F2RL1,,1921,75,-0.129,0.307,0.000806,-0.0525,0.0216,0.00272,0.0068,0.00128,0,0.0132,1,True,7.0
TNFRSF14,,1920,75,-0.129,0.333,0.0122,0.193,0.0147,0.295,0.259,0.107,0.974,0.84,2.46e-05,True,1.0
OR4D1,,1918,75,-0.127,0.2,0.00231,-0.0646,0.00781,0.000295,0.00509,0,0,0.0103,1,True,7.0
CD40,,1917,75,-0.125,0.253,0.000358,-0.0465,0.0155,0.0049,0.0075,0.00226,0.0128,0.09,0.429,True,1.0
DIP2A,,1914,75,-0.121,0.373,0.01,0.167,0.0246,0.142,0.123,0.0503,0.692,0.823,2.17e-05,True,
LILRB1,,1912,75,-0.119,0.16,2.13e-05,-0.0518,8.97e-05,0.000305,0.00532,0,0.0256,0.00598,0.875,True,1.0
KLRC2,,1908,75,-0.114,0.04,8.89e-07,-0.0515,2.67e-06,0,0.00634,0.0125,0,-0.503,8.22e-05,True,1.0
JAM3,,1907,75,-0.111,0.267,0.0154,-0.0432,0.0861,0.00182,0.00682,0.00128,0.0385,-0.0104,0.846,True,1.0
ADAM28,,1903,75,-0.108,0.187,0.00142,-0.0486,0.0109,0.000322,0.00282,0,0,0.0109,0.00316,True,1.0
NMUR1,,1899,75,-0.0988,0.0267,2.81e-08,-0.0803,1.68e-08,0,0.00284,0,0,0,0.508,True,7.0
LGR6,,1898,75,-0.0979,0.107,3.06e-07,-0.0459,3.7e-07,0,0.00287,0,0,0,1,True,7.0
B2M,,1893,75,-0.0938,0.187,8.42e-08,-0.00157,0.0124,0.992,0.999,0.994,1,-0.062,0.869,True,
ABCA1,,1892,75,-0.093,0.24,6.19e-05,-0.0488,0.000223,0.00149,0.00578,0,0.0128,0.0755,0.0163,True,15.0
CD248,,1890,75,-0.0904,0.173,0.0023,-0.0422,0.0104,0.000184,0.0044,0,0.0256,0.00467,0.199,True,1.0
CDCP1,,1889,75,-0.0901,0.28,0.0295,-0.0345,0.111,0.00201,0.00402,0,0.0385,0.0716,0.715,True,1.0
AMIGO1,,1888,75,-0.0901,0.2,0.000479,-0.0376,0.0137,0.00285,0.00518,0,0,0.0765,0.00561,True,1.0
TTYH2,,1884,75,-0.0865,0.347,0.0328,-0.0245,0.546,0.00694,0.00995,0,0,0.203,0.0074,True,5.0
SIGLEC7,,1877,75,-0.0764,0.12,7.79e-05,-0.0487,0.000209,0,0.0032,0,0.0641,0,0.745,True,1.0
ACE,,1875,75,-0.0759,0.293,0.00527,-0.0412,0.022,0.000354,0.00406,0,0.0256,0.0111,0.519,True,1.0
IL17RB,,1873,75,-0.0748,0.28,0.0211,-0.0198,0.171,0.000995,0.0036,0.0112,0.0256,-0.387,0.11,True,1.0
ROBO3,,1872,75,-0.0745,0.28,0.00509,-0.0391,0.164,0.00107,0.00366,0,0,0.0273,0.0577,True,1.0
CNR2,,1871,75,-0.0723,0.16,0.000949,-0.0444,0.000742,0.000348,0.00324,0,0.0128,0.0117,0.46,True,7.0
TGFA,,1867,75,-0.0701,0.2,3.44e-05,-0.0365,0.00011,0.000689,0.00267,0,0,0.0223,0.84,True,1.0
SLC4A4,,1864,75,-0.0689,0.107,4.24e-07,-0.0326,7.47e-07,0,0.00184,0,0,0,0.836,True,10.0
ADAM23,,1859,75,-0.0637,0.227,0.00937,-0.0151,0.149,0.0003,0.00308,0,0.0128,0.00893,0.689,True,1.0
PDGFRB,,1858,75,-0.0631,0.2,1.46e-05,-0.0255,0.0001,0.0005,0.00307,0.00128,0,-0.032,0.592,True,1.0
TLR1,,1855,75,-0.0618,0.373,0.0382,0,0.628,0.0058,0.00783,0,0.0128,0.211,0.00327,True,1.0
MMRN1,,1852,75,-0.0575,0.24,0.0328,-0.0255,0.173,0.000813,0.002,0,0.0256,0.0282,0.41,True,
LTBR,,1847,75,-0.0519,0.253,0.000862,-0.0163,0.0342,0.00147,0.00333,0,0,0.0433,0.274,True,1.0
A2M,,1844,75,-0.0501,0.293,0.000521,0,0.0531,0.0034,0.00501,0.00767,0,-0.109,0.758,True,
ADGRG3,,1843,75,-0.0469,0.12,5.66e-06,-0.0195,1.47e-05,0,0.00148,0,0,0,0.0192,True,7.0
TNFRSF13B,,1841,75,-0.0457,0.227,0.000251,-0.0161,0.0064,0.000328,0.00205,0,0.0128,0.014,0.159,True,1.0
GPR160,,1840,75,-0.0429,0.333,0.037,-0.0111,0.467,0.00186,0.00295,0,0,0.0608,0.0006,True,7.0
PROCR,,1836,75,-0.0391,0.267,0.00726,-0.0136,0.0439,0.00117,0.00229,0,0,0.0352,0.0599,True,1.0
CHRNE,,1835,75,-0.0385,0.32,0.0123,0,0.268,0.00162,0.00592,0.00683,0,-0.234,0.434,True,4.0
TIE1,,1834,75,-0.038,0.28,0.0136,-0.0117,0.116,0.000517,0.00175,0,0.0128,0.0205,0.0696,True,1.0
FCGR3B,,1832,75,-0.0367,0.12,6.62e-06,-0.0208,1.27e-05,0,0.00124,0,0.0128,0,0.738,True,0.0
GPR141,,1830,75,-0.0363,0.133,3.5e-05,-0.0381,0.000173,0,0.00182,0,0.0128,0,1,True,7.0
COL5A3,,1823,75,-0.0302,0.267,0.00896,-0.00807,0.016,0.00106,0.00143,0,0,0.0397,0.00817,True,
PLXNA4,,1818,75,-0.0281,0.187,0.00712,-0.0131,0.0347,9.59e-05,0.00197,0,0.0128,0.00471,0.169,True,1.0
TLR3,,1814,75,-0.0268,0.173,0.000151,-0.0118,0.00357,0.000325,0.00178,0,0,0.00635,0.0152,True,1.0
DCN,,1812,75,-0.0264,0.307,0.0495,0,0.605,0.00969,0.00797,0.0339,0.115,-1.01,0.0558,True,
TNXB,,1810,75,-0.0259,0.307,0.0232,0,0.634,0.00177,0.00168,0,0.0128,0.0519,0.188,True,
CD33,,1809,75,-0.0255,0.147,3.66e-05,-0.0193,5.82e-05,0,0.0014,0,0.0128,0,0.689,True,1.0
ADGRE2,,1803,75,-0.0222,0.32,0.00435,-0.00646,0.0288,0.00064,0.00141,0,0.0128,0.026,0.34,True,7.0
BMP2,,1802,75,-0.0206,0.133,6.02e-05,0,0.000367,0,0.00112,0,0,0,1,True,
DCHS1,,1799,75,-0.0173,0.253,0.0431,0,0.222,0.000548,0.00138,0,0,0.0176,0.0261,True,1.0
SULF2,,1796,75,-0.0157,0.333,0.0429,0,1,0.00253,0.00196,0,0,0.0694,0.741,True,
DSC2,,1793,75,-0.0135,0.28,0.00486,0,0.295,0.00214,0.00194,0,0,0.0898,0.0571,True,1.0
SLCO4C1,,1790,75,-0.013,0.24,0.0151,0,0.0947,0,0.00178,0,0,0,0.124,True,12.0
CXCR2,,1785,75,-0.0111,0.173,5.49e-06,0,1.29e-05,0,0.000632,0,0,0,0.562,True,7.0
CMKLR1,,1781,75,-0.00982,0.2,0.00257,0,0.024,0,0.000577,0,0,0,0.0474,True,7.0
GP5,,1778,75,-0.00962,0.12,0.000242,0,0.000573,0,0.000496,0,0,0,0.448,True,1.0
DSC3,,1772,75,-0.00848,0.267,0.00497,0,0.386,0.00525,0.00397,0.0135,0.0897,-0.314,0.745,True,1.0
TMPRSS13,,1769,75,-0.00429,0.267,0.0172,0,0.536,0.000645,0.000956,0,0,0.0216,0.0164,True,1.0
F2RL2,,1768,75,-0.00418,0.32,0.0454,0,0.97,0.00323,0.00294,0,0,0.0925,0.177,True,7.0
CEACAM1,,1767,75,-0.00299,0.347,0.0412,0,0.313,0.00144,0.00267,0,0.0385,0.0511,0.00669,True,1.0
CR1,,1766,75,-0.00269,0.147,0.00244,0,0.0115,0,0.000211,0,0,0,0.00602,True,1.0
NECTIN2,,1764,75,-0.0018,0.32,0.0483,0,1,0.00304,0.00268,0,0,0.106,0.27,True,
HAVCR1,,1763,75,-0.00177,0.16,0.000246,0,0.00102,0,0.00055,0,0,0,0.00583,True,1.0
SIGLEC10,,1759,75,-7.16e-05,0.2,0.00261,0,0.0203,0,0.000519,0,0,0,0.228,True,1.0
```

## §3d Lineage controls (contamination reference)

**What:** the same statistics for transcripts a CD4 T cell does not make. If these are up on malignant CD4, part of any malignant-up signal can be contamination of that size.

**How:** same tests as §3b. 42 §2 turns this reference into formal per-gene artifact flags; this notebook only reports it.

```csv
gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm
SFN,,,75,0.0455,0.56,0.29,0.0768,0.000426,0.0634,0.0283,0.0625,0.423,-0.0601,0.349,False,
CD68,,421,75,0.0346,0.533,0.448,0.079,0.02,0.0208,0.0137,0.00559,0.0769,0.44,0.00166,True,1.0
KRT10,,,75,0.0261,0.533,1,0.181,5.77e-06,0.408,0.361,0.285,0.949,0.303,0.237,False,
EGFL7,,,75,0.0216,0.56,0.121,0.0268,0.00529,0.00455,0.00192,0,0.0256,0.116,0.274,False,
COL1A1,,468,75,0.0153,0.533,0.194,0.105,1.12e-05,0.0357,0.0117,0.0227,0.167,-0.0338,0.298,True,
KRT17,,,75,0.0066,0.533,0.72,0.0197,0.0307,0.0156,0.00597,0.0083,0.154,0.143,0.3,False,
C1QB,,,75,0.00442,0.507,0.466,0.0344,0.00186,0.00652,0.00267,0,0.0256,0.218,7.31e-06,False,
CD14,,522,75,0,0.453,0.21,0.00951,0.545,0.00241,0.00274,0,0,0.0876,0.0169,True,0.0
SFRP2,,,75,0,0.347,0.689,0,0.547,0.00108,0.000792,0,0.0513,0.041,1,False,
VWF,,522,75,0,0.427,0.842,0,0.23,0.000995,0,0,0,0.0312,0.157,True,
TYRP1,,522,75,0,0.293,0.106,0,1,0.00117,0.000964,0,0.0128,0.0424,0.689,True,1.0
PMEL,,,75,0,0.493,0.66,0.0138,0.00223,0.00269,0.000975,0,0,0.0752,0.0343,False,1.0
HBB,,,75,0,0.333,0.371,0,0.702,0.00597,0.00896,0.0124,0.128,-0.227,0.369,False,
HBA2,,,75,0,0.333,0.424,0,0.908,0.00127,0.00079,0,0.0128,0.0609,0.802,False,
CPA3,,,75,0,0.267,0.527,0,0.804,0,0,0,0,0,0.452,False,
COL1A2,,522,75,0,0.373,0.563,0,0.173,0.00469,0.00207,0.0142,0.0897,-0.503,0.32,True,
DCT,,522,75,0,0.387,0.974,0,0.339,0.000267,0,0,0,0.00888,0.94,True,1.0
LUM,,522,75,0,0.307,0.534,0,0.638,0.00118,0.00145,0,0,0.034,0.0836,True,
CDH5,,522,75,0,0.2,0.21,0,0.767,0,0,0,0,0,0.27,True,1.0
MS4A2,,,75,0,0.253,0.858,0,0.161,0,0,0,0,0,0.0367,False,
CD163,,522,75,0,0.333,0.701,0,0.463,0.000373,0,0,0,0.0177,0.00202,True,1.0
MLANA,,,75,0,0.293,0.0568,0,0.565,0.000322,0.000516,0,0,0.0102,0.214,False,
TPSAB1,,,75,0,0.227,0.903,0,0.106,0,0,0,0,0,0.766,False,
TPSB2,,,75,0,0.187,0.181,0,0.854,0,0,0,0,0,0.798,False,
CLDN5,,,75,0,0.493,0.0126,0.0313,0.000735,0.0037,0.0023,0,0.0769,0.146,0.0846,False,
KRTDAP,,,75,0,0.4,0.14,0,0.872,0.0148,0.0116,0.0612,0.205,-1.3,0.342,False,
JCHAIN,,,75,0,0.347,0.938,0,0.465,0.000824,0.000515,0,0.0128,0.0412,0.0424,False,
COL3A1,,1761,75,-0.000764,0.307,0.134,0,0.358,0.00459,0.00273,0.0112,0.128,-0.241,0.689,True,
KRT6A,,,75,-0.0112,0.32,0.0141,0,0.743,0.0109,0.0106,0.00218,0.308,0.278,0.0198,False,
C1QA,,,75,-0.0161,0.413,0.16,0,0.933,0.00584,0.00351,0,0.0256,0.239,0.00907,False,
DCN,,1812,75,-0.0264,0.307,0.0495,0,0.605,0.00969,0.00797,0.0339,0.115,-1.01,0.0558,True,
KRT5,,,75,-0.052,0.36,0.0107,0.0121,0.19,0.0367,0.036,0.0625,0.333,-0.763,0.622,False,
LYZ,,,75,-0.0733,0.387,0.0328,-0.0131,1,0.0171,0.0124,0,0.167,0.562,0.000841,False,
CD79A,,1885,75,-0.0873,0.347,0.413,-0.0449,0.819,0.00509,0.00986,0,0.0513,0.148,0.00113,True,1.0
DMKN,,,75,-0.0882,0.293,0.00372,-0.0178,0.872,0.0148,0.0156,0.0894,0.256,-1.49,0.0561,False,
KRT14,,,75,-0.0946,0.36,0.00379,0.00822,0.398,0.07,0.0449,0.193,0.397,-1.42,0.343,False,
MZB1,,,75,-0.106,0.427,0.893,-0.00795,0.406,0.00798,0.0128,0,0.141,0.223,0.000653,False,
KRT1,,1940,75,-0.152,0.28,0.00439,-0.0028,0.737,0.0261,0.024,0.129,0.397,-1.68,0.238,True,
COL6A2,,1954,75,-0.17,0.373,0.635,-0.0104,0.329,0.0262,0.0254,0.0174,0.179,0.226,0.254,True,
MS4A1,,1972,75,-0.203,0.187,2.06e-05,-0.0794,0.000183,0.00187,0.0104,0.0134,0.0128,-0.457,0.0514,True,4.0
```

## §4a Positive controls

**What:** where the four clinically validated CTCL antigens land in the ranking.

**How:** rank of log2fc_paired among 2,115 tested surface genes; rank_pct = rank / 2,115. Pass = all four near the top (well inside the top ~10%).

```csv
gene,cd_name,agent,stage,rank_surface,rank_pct,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,frac_donors_up_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc
TNFRSF8,CD30,brentuximab vedotin; AFM13 (CD30xCD16A),approved,99,0.0468,75,0.332,0.667,0.000215,0.455,0.747,4.36e-07,0.038,0.0111,0,0.192,0.86,1.05e-07
CCR4,CCR4,mogamulizumab,approved,101,0.0478,75,0.324,0.56,0.0909,0.568,0.613,0.00126,0.219,0.135,0.0656,0.705,1.12,0.000897
KIR3DL2,CD158k,lacutamab (IPH4102),phase 2,522,0.247,75,0,0.36,0.000947,0,0.36,0.000438,0,0.00372,0.00256,0.321,-0.108,0.678
IL2RA,CD25,denileukin diftitox-cxdl (Lymphir); camidanlumab tesirine,approved (2024-08),1956,0.925,75,-0.172,0.387,0.454,0.0171,0.507,0.214,0.122,0.126,0.0828,0.628,0.16,0.153
```

## §4b Loss markers

**What:** the CTCL loss phenotype (CD7, DPP4/CD26, CD5, CD6, CD27, CD2) on malignant vs reactive CD4.

**How:** same paired test; CD7 and DPP4 should be negative with a small q. A missing gene is absent from the tables (CD26 is DPP4).

```csv
gene,cd_name,agent,stage,rank_surface,rank_pct,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,frac_donors_up_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc
CD7,CD7,WU-CART-007; BEAM-201 (base-edited),phase 1/2,2115,1,75,-2.85,0.0933,3.6e-09,-2.28,0.133,2.59e-10,0.038,0.539,0.211,0.308,-2.06,0.0526
DPP4,,,,2090,0.988,75,-0.841,0.253,0.00391,-0.575,0.307,0.0257,0.0087,0.0457,0.0492,0.179,-1.15,0.267
CD27,,,,2022,0.956,75,-0.359,0.467,0.792,-0.013,0.493,0.315,0.246,0.278,0.0816,0.59,1.15,0.0218
CD2,,,,1931,0.913,75,-0.146,0.453,0.0478,0.0676,0.533,1,0.6,0.583,0.392,0.897,0.278,0.00618
CD5,CD5,CD5 CAR-T; MT-101 (CAR-monocyte),phase 1/2,1854,0.877,75,-0.0584,0.493,0.816,0.173,0.587,0.177,0.31,0.262,0.104,0.744,1.09,0.00268
CD6,,,,327,0.155,75,0.0957,0.56,0.344,0.323,0.667,0.00331,0.395,0.349,0.183,0.885,0.982,0.00164
```

## §5a Shortlist funnel

**What:** how many genes survive each gate, applied cumulatively in the order listed.

**How:** gates as stated in the markdown above.

```csv
gate,n_genes_remaining
all genes in the tables,2355
surface gene,2115
"not lineage control, not loss marker",2093
q < 0.05,817
log2fc_paired > 0.5,33
malig_frac_donors_pos >= 0.5,23
```

## §5b Shortlist

**What:** surface genes selectively up on malignant CD4, present in at least half of patients.

**How:** all five gates; sorted by log2fc_paired. hc_* columns = expression on healthy-skin CD4, i.e. the T-cell on-target off-tumour cost. disputed_in_42 = one of the genes 42 names as implausible surface targets (cytoplasmic, ER, mitochondrial, secreted or stromal protein) despite passing the statistics.

**Full table on disk:** `tables/tgt49_shortlist.csv`

```csv
gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm,agent,stage,hc_frac_donors_pos,react_frac_donors_pos,n_evidence,topology,surfy_class,source,surface_caveat,disputed_in_42
CD70,CD70,1,75,1.1,0.92,3.09e-10,1.22,4.71e-11,0.21,0.0498,0.024,0.808,1.91,1.05e-07,True,1.0,INCA036873 (CD70xCD3); CTX130/CTX131,phase 1,0,0.195,2,CY:1-20;TM:21-42;NC:43-193,Unclassified,both,,False
CD9,,2,75,1.08,0.733,7.09e-07,1.12,1.55e-08,0.25,0.0583,0.107,0.718,0.771,0.0172,True,4.0,,,0.524,0.329,3,CY:1-11;TM:12-35;NC:36-54;TM:55-78;CY:79-88;TM:89-111;NC:112-194;TM:195-220;CY:221-228,Miscellaneous,surfaceome,,False
CPM,,3,75,1.02,0.747,7.08e-08,0.954,6.44e-09,0.137,0.0248,0.00669,0.577,1.89,4.34e-06,True,0.0,,,0,0.0732,3,SP:1-17;NC:18-443,Enzymes,surfaceome,,False
COL6A3,,6,75,0.811,0.88,3.28e-09,0.843,1.5e-10,0.113,0.0247,0.0158,0.564,1.47,2.98e-06,True,,,,0,0.134,1,,,surfaceome,,True
CD151,,7,75,0.794,0.76,9.39e-08,0.946,1.77e-10,0.196,0.0889,0.0612,0.795,1.04,4.47e-05,True,4.0,,,0.19,0.415,3,CY:1-20;TM:21-43;NC:44-57;TM:58-81;CY:82-91;TM:92-115;NC:116-221;TM:222-245;CY:246-253,Miscellaneous,surfaceome,,False
GRN,,9,75,0.778,0.907,9.81e-10,0.835,6.51e-11,0.211,0.103,0.041,0.846,1.58,2.29e-07,True,,,,0.0952,0.524,1,,,surfaceome,,True
TFRC,,11,75,0.731,0.893,1.67e-09,0.806,4.71e-11,0.296,0.147,0.118,0.949,0.806,4.47e-05,True,,,,0.571,0.659,2,,,both,,False
SLC39A6,,12,75,0.707,0.947,1.91e-10,0.897,4.71e-11,0.183,0.0789,0.0447,0.91,1.2,2.11e-07,True,6.0,,,0.0476,0.256,4,SP:1-26;NC:27-322;TM:323-346;CY:347-353;TM:354-377;NC:378-424;TM:425-444;CY:445-657;TM:658-679;NC:680-684;TM:685-703;CY:704-723;TM:724-745;NC:746-755,Transporters,surfaceome,,False
SLC4A2,,13,75,0.698,0.933,1.91e-10,0.739,4.71e-11,0.13,0.0406,0.0156,0.641,1.51,1.05e-07,True,,,,0,0.134,2,,,surfaceome,,False
TNFRSF25,,14,75,0.698,0.747,9.14e-05,0.821,1.79e-07,0.339,0.171,0.0943,0.872,1.31,2.13e-06,True,1.0,,,0.333,0.768,3,SP:1-24;NC:25-200;TM:201-223;CY:224-417,Receptors,surfaceome,,False
SLC1A4,,15,75,0.69,0.747,5.23e-05,0.682,1.64e-07,0.144,0.0615,0.0239,0.628,1.38,2.59e-06,True,9.0,,,0.0476,0.317,3,CY:1-42;TM:43-65;NC:66-84;TM:85-105;CY:106-116;TM:117-139;NC:140-219;TM:220-237;CY:238-257;TM:258-277;NC:278-296;TM:297-320;CY:321-327;TM:328-349;NC:350-368;TM:369-394;CY:395-414;TM:415-438;NC:439-532,Transporters,surfaceome,,False
LGALS3,,16,75,0.663,0.587,0.00115,0.631,3.82e-05,0.427,0.292,0.361,0.885,0.21,0.184,True,,,,0.81,0.963,2,,,surfaceome,,True
CD82,,17,75,0.632,0.773,1.41e-07,0.714,2.06e-09,0.477,0.283,0.176,1,1.04,1.32e-05,True,4.0,,,0.667,0.988,2,CY:1-11;TM:12-36;NC:37-55;TM:56-75;CY:76-84;TM:85-108;NC:109-228;TM:229-253;CY:254-267,Miscellaneous,both,,False
SLC5A3,,19,75,0.617,0.893,1.81e-09,0.719,4.71e-11,0.217,0.0912,0.0499,0.769,1.45,7.22e-07,True,12.0,,,0.0952,0.439,3,NC:1-11;TM:12-29;CY:30-39;TM:40-61;NC:62-81;TM:82-102;CY:103-121;TM:122-143;NC:144-156;TM:157-182;CY:183-188;TM:189-209;NC:210-301;TM:302-323;CY:324-357;TM:358-376;NC:377-408;TM:409-432;CY:433-443;TM:444-462;NC:463-510;TM:511-532;CY:533-695;TM:696-716;NC:717-718,Transporters,surfaceome,,False
IL6R,,21,75,0.614,0.72,2.31e-05,0.778,1.51e-08,0.233,0.107,0.0726,0.769,0.857,0.000122,True,1.0,,,0.381,0.524,2,SP:1-19;NC:20-364;TM:365-388;CY:389-468,Receptors,surfaceome,,False
SYPL1,,22,75,0.599,0.907,1.15e-09,0.807,5.78e-11,0.362,0.2,0.179,0.949,0.674,0.000435,True,4.0,,,0.667,0.878,2,CY:1-34;TM:35-55;NC:56-116;TM:117-139;CY:140-150;TM:151-171;NC:172-210;TM:211-233;CY:234-259,Miscellaneous,surfaceome,,True
CALU,,23,75,0.587,0.88,6.18e-10,0.801,4.71e-11,0.19,0.0944,0.0392,0.846,1.26,5.31e-07,True,,,,0.19,0.415,1,,,surfaceome,,True
QSOX1,,24,75,0.584,0.853,7.51e-09,0.714,1.06e-10,0.133,0.0677,0.0261,0.692,1.23,2.32e-07,True,1.0,,,0,0.293,2,SP:1-29;NC:30-705;TM:706-731;CY:732-747,Enzymes,surfaceome,,True
VDAC1,,25,75,0.584,0.947,1.91e-10,0.645,4.71e-11,0.482,0.293,0.175,0.987,1.28,5.15e-07,True,,,,0.762,0.951,1,,,surfaceome,,True
TNFRSF4,,28,75,0.542,0.667,0.00308,0.538,1.81e-05,0.349,0.223,0.137,0.897,1.32,0.000118,True,1.0,,,0.762,0.89,2,SP:1-28;NC:29-213;TM:214-240;CY:241-277,Receptors,both,,False
TSPAN2,,29,75,0.53,0.747,1.24e-05,0.726,2.12e-08,0.135,0.0534,0.0377,0.654,0.841,0.000873,True,4.0,,,0.0952,0.159,2,CY:1-11;TM:12-35;NC:36-54;TM:55-78;CY:79-86;TM:87-111;NC:112-187;TM:188-213;CY:214-221,Miscellaneous,surfaceome,,False
GPI,,32,75,0.507,0.827,2.79e-07,0.618,7.24e-10,0.474,0.303,0.125,0.974,1.33,2.55e-05,True,,,,0.619,0.939,2,,,surfaceome,,True
TMEM30A,,33,75,0.506,0.787,1.08e-08,0.686,4.71e-11,0.188,0.104,0.0612,0.846,1.04,9.01e-06,True,2.0,,,0.19,0.549,2,CY:1-43;TM:44-69;NC:70-324;TM:325-346;CY:347-361,Miscellaneous,surfaceome,,False
```

## §5c Shortlist genes with no clinical precedent

**What:** the shortlist minus every gene with an entry in surface_targets.CLINICAL_PRECEDENT.

**How:** stage column empty = no approved, trial or preclinical record curated in this project; that is a statement about this project's curation, not a literature search.

```csv
gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm,agent,stage,hc_frac_donors_pos,react_frac_donors_pos,n_evidence,topology,surfy_class,source,surface_caveat,disputed_in_42
CD9,,2,75,1.08,0.733,7.09e-07,1.12,1.55e-08,0.25,0.0583,0.107,0.718,0.771,0.0172,True,4.0,,,0.524,0.329,3,CY:1-11;TM:12-35;NC:36-54;TM:55-78;CY:79-88;TM:89-111;NC:112-194;TM:195-220;CY:221-228,Miscellaneous,surfaceome,,False
CPM,,3,75,1.02,0.747,7.08e-08,0.954,6.44e-09,0.137,0.0248,0.00669,0.577,1.89,4.34e-06,True,0.0,,,0,0.0732,3,SP:1-17;NC:18-443,Enzymes,surfaceome,,False
COL6A3,,6,75,0.811,0.88,3.28e-09,0.843,1.5e-10,0.113,0.0247,0.0158,0.564,1.47,2.98e-06,True,,,,0,0.134,1,,,surfaceome,,True
CD151,,7,75,0.794,0.76,9.39e-08,0.946,1.77e-10,0.196,0.0889,0.0612,0.795,1.04,4.47e-05,True,4.0,,,0.19,0.415,3,CY:1-20;TM:21-43;NC:44-57;TM:58-81;CY:82-91;TM:92-115;NC:116-221;TM:222-245;CY:246-253,Miscellaneous,surfaceome,,False
GRN,,9,75,0.778,0.907,9.81e-10,0.835,6.51e-11,0.211,0.103,0.041,0.846,1.58,2.29e-07,True,,,,0.0952,0.524,1,,,surfaceome,,True
TFRC,,11,75,0.731,0.893,1.67e-09,0.806,4.71e-11,0.296,0.147,0.118,0.949,0.806,4.47e-05,True,,,,0.571,0.659,2,,,both,,False
SLC39A6,,12,75,0.707,0.947,1.91e-10,0.897,4.71e-11,0.183,0.0789,0.0447,0.91,1.2,2.11e-07,True,6.0,,,0.0476,0.256,4,SP:1-26;NC:27-322;TM:323-346;CY:347-353;TM:354-377;NC:378-424;TM:425-444;CY:445-657;TM:658-679;NC:680-684;TM:685-703;CY:704-723;TM:724-745;NC:746-755,Transporters,surfaceome,,False
SLC4A2,,13,75,0.698,0.933,1.91e-10,0.739,4.71e-11,0.13,0.0406,0.0156,0.641,1.51,1.05e-07,True,,,,0,0.134,2,,,surfaceome,,False
TNFRSF25,,14,75,0.698,0.747,9.14e-05,0.821,1.79e-07,0.339,0.171,0.0943,0.872,1.31,2.13e-06,True,1.0,,,0.333,0.768,3,SP:1-24;NC:25-200;TM:201-223;CY:224-417,Receptors,surfaceome,,False
SLC1A4,,15,75,0.69,0.747,5.23e-05,0.682,1.64e-07,0.144,0.0615,0.0239,0.628,1.38,2.59e-06,True,9.0,,,0.0476,0.317,3,CY:1-42;TM:43-65;NC:66-84;TM:85-105;CY:106-116;TM:117-139;NC:140-219;TM:220-237;CY:238-257;TM:258-277;NC:278-296;TM:297-320;CY:321-327;TM:328-349;NC:350-368;TM:369-394;CY:395-414;TM:415-438;NC:439-532,Transporters,surfaceome,,False
LGALS3,,16,75,0.663,0.587,0.00115,0.631,3.82e-05,0.427,0.292,0.361,0.885,0.21,0.184,True,,,,0.81,0.963,2,,,surfaceome,,True
CD82,,17,75,0.632,0.773,1.41e-07,0.714,2.06e-09,0.477,0.283,0.176,1,1.04,1.32e-05,True,4.0,,,0.667,0.988,2,CY:1-11;TM:12-36;NC:37-55;TM:56-75;CY:76-84;TM:85-108;NC:109-228;TM:229-253;CY:254-267,Miscellaneous,both,,False
SLC5A3,,19,75,0.617,0.893,1.81e-09,0.719,4.71e-11,0.217,0.0912,0.0499,0.769,1.45,7.22e-07,True,12.0,,,0.0952,0.439,3,NC:1-11;TM:12-29;CY:30-39;TM:40-61;NC:62-81;TM:82-102;CY:103-121;TM:122-143;NC:144-156;TM:157-182;CY:183-188;TM:189-209;NC:210-301;TM:302-323;CY:324-357;TM:358-376;NC:377-408;TM:409-432;CY:433-443;TM:444-462;NC:463-510;TM:511-532;CY:533-695;TM:696-716;NC:717-718,Transporters,surfaceome,,False
IL6R,,21,75,0.614,0.72,2.31e-05,0.778,1.51e-08,0.233,0.107,0.0726,0.769,0.857,0.000122,True,1.0,,,0.381,0.524,2,SP:1-19;NC:20-364;TM:365-388;CY:389-468,Receptors,surfaceome,,False
SYPL1,,22,75,0.599,0.907,1.15e-09,0.807,5.78e-11,0.362,0.2,0.179,0.949,0.674,0.000435,True,4.0,,,0.667,0.878,2,CY:1-34;TM:35-55;NC:56-116;TM:117-139;CY:140-150;TM:151-171;NC:172-210;TM:211-233;CY:234-259,Miscellaneous,surfaceome,,True
CALU,,23,75,0.587,0.88,6.18e-10,0.801,4.71e-11,0.19,0.0944,0.0392,0.846,1.26,5.31e-07,True,,,,0.19,0.415,1,,,surfaceome,,True
QSOX1,,24,75,0.584,0.853,7.51e-09,0.714,1.06e-10,0.133,0.0677,0.0261,0.692,1.23,2.32e-07,True,1.0,,,0,0.293,2,SP:1-29;NC:30-705;TM:706-731;CY:732-747,Enzymes,surfaceome,,True
VDAC1,,25,75,0.584,0.947,1.91e-10,0.645,4.71e-11,0.482,0.293,0.175,0.987,1.28,5.15e-07,True,,,,0.762,0.951,1,,,surfaceome,,True
TNFRSF4,,28,75,0.542,0.667,0.00308,0.538,1.81e-05,0.349,0.223,0.137,0.897,1.32,0.000118,True,1.0,,,0.762,0.89,2,SP:1-28;NC:29-213;TM:214-240;CY:241-277,Receptors,both,,False
TSPAN2,,29,75,0.53,0.747,1.24e-05,0.726,2.12e-08,0.135,0.0534,0.0377,0.654,0.841,0.000873,True,4.0,,,0.0952,0.159,2,CY:1-11;TM:12-35;NC:36-54;TM:55-78;CY:79-86;TM:87-111;NC:112-187;TM:188-213;CY:214-221,Miscellaneous,surfaceome,,False
GPI,,32,75,0.507,0.827,2.79e-07,0.618,7.24e-10,0.474,0.303,0.125,0.974,1.33,2.55e-05,True,,,,0.619,0.939,2,,,surfaceome,,True
TMEM30A,,33,75,0.506,0.787,1.08e-08,0.686,4.71e-11,0.188,0.104,0.0612,0.846,1.04,9.01e-06,True,2.0,,,0.19,0.549,2,CY:1-43;TM:44-69;NC:70-324;TM:325-346;CY:347-361,Miscellaneous,surfaceome,,False
```

## §6a Per-donor log2FC, malignant vs reactive CD4

**What:** the paired effect inside every patient, for each detail gene.

**How:** per donor: log2((mean_all_malig + 0.05) / (mean_all_react + 0.05)), mean_all = mean log1p CP10K over the arm's cells; donors ordered by burden, low to high. NaN = gene not tested in that donor.

**Full table on disk:** `tables/tgt49_donor_lfc.csv`

```csv
donor,study,dx2,lesion,clin_stage_class,burden,n_malignant,n_reactive,CD70,CD9,CPM,COL6A3,CD151,GRN,TFRC,SLC39A6,SLC4A2,TNFRSF25,SLC1A4,LGALS3,CD82,SLC5A3,IL6R,SYPL1,CALU,QSOX1,VDAC1,TNFRSF4,TSPAN2,GPI,TMEM30A,CCR4,TNFRSF8,KIR3DL2,IL2RA,CD7,DPP4,CD5,CD6,CD27,CD2
Li2024_atlas__MF17,li2024,MF,tumor,advanced,0.00837,37,4385,0.958,0,0.842,0.279,0.167,0.32,-0.207,0.397,0.529,-0.309,1.45,0.372,0.0137,-0.603,-0.553,0.141,0.617,0.744,1.59,-1.61,0.228,0.837,0.16,-0.852,1.61,0,-0.157,-2.23,-0.177,-0.206,-0.21,-0.515,-0.277
Li2024_atlas__CTCL7,li2024,MF,mixed,early,0.0869,3658,38432,1.66,4.83,0.45,0.246,1.79,0.75,-0.604,2.66,0.697,-1.75,2.21,1.85,1.62,0.743,-1.29,0.0516,-0.303,1.25,0.282,-0.629,-0.886,0.00171,0.661,-2.78,0.477,0,0.114,-3.12,0.194,-0.958,-0.761,-1.57,0.995
Li2024_atlas__MF27,li2024,MF,tumor,advanced,0.129,107,725,3.23,0.864,0.448,1.91,1.29,1.3,0.704,0.702,1.44,-1.34,1.41,1.2,0.78,0.479,-0.467,0.134,0.619,0.756,1.11,0.0975,0.789,0.825,0.147,-0.165,1.92,0,0.57,-1.01,-1.25,0.0552,-0.0978,0.583,0.143
Li2024_atlas__CTCL4,li2024,MF,mixed,early,0.136,1241,7906,2.64,3.3,0.539,0.194,0.077,0.679,0.824,2.25,0.435,0.0293,2.66,1.65,0.595,1.5,-0.338,0.346,0.376,0.56,0.255,-1.32,0.339,0.29,0.206,-1.2,1.08,0,-0.331,-2.01,-0.484,-0.723,0.763,-1.53,0.378
D15__MF28,il4ra2026,MF,tumor,advanced,0.151,123,691,2.41,2.73,1.68,0.706,-0.0346,-0.502,0.672,-0.293,0.264,-1.01,0.842,1.34,-0.721,1.14,1.08,0.192,0.478,0.761,0.00329,-0.914,0.156,-0.523,0.448,-0.113,0.503,-0.296,0.32,-2.85,-0.232,-2.01,-2.26,-1.11,-0.859
D10__JONAK1,jonak2021,MF,plaque,early,0.153,348,1920,0.841,1.34,1.39,1.58,1.31,0.499,1.07,0.847,2.06,1.47,0.135,-0.693,1.24,0.846,0.843,0.493,0.833,0.307,0.959,1.57,1.66,0.912,0.506,0.434,0.124,2.33,-0.427,-2.76,-0.865,1.53,1.47,0.576,-0.321
D13__MF21,gaydosik2022,MF,tumor,advanced,0.177,118,547,1.16,-0.854,3.14,-0.284,1.31,0.923,1.66,1.07,2.34,0.384,2.25,-0.714,0.619,-0.0928,-0.69,2.6,2.06,1.38,1.6,-0.76,0.152,1.67,0.0151,2.7,-0.245,3.33,1.33,-2.48,-0.755,0.875,-0.794,1.78,-0.146
D3__P105,brunner2024,MF,erythroderma,advanced,0.185,117,516,0.485,-0.163,1.2,0.156,-0.385,1.11,-0.887,-0.216,0.439,0.209,-0.835,-0.631,0.566,0.738,0.413,0.515,-0.537,-0.326,0.185,0.616,2.18,-0.896,0.576,1.52,-1.22,1.32,-0.172,-2.6,-0.337,0.202,-0.312,0.121,-0.436
D15__MF36,il4ra2026,MF,mixed,advanced,0.221,620,2182,0.105,2.8,1.59,1.62,1.17,0.434,0.341,0.938,0.412,0.754,-0.17,-0.393,0.768,0.661,0.329,0.844,0.444,0.584,0.895,-0.385,1.45,0.787,0.536,-0.113,-0.0519,3.2,-1.3,-2.65,-0.826,-0.491,0.308,-0.504,0.0733
D8__P90,alkon2024,MF,patch,early,0.227,68,232,1.18,1.58,-0.253,0.529,-0.535,-1.03,-0.0237,0.387,0.345,0.0701,1.66,1.44,-0.145,0.36,-1.56,0.17,-0.0641,-0.0589,0.485,-3.13,0.53,0.337,0.653,-2.3,0.983,-0.233,-0.0703,-3.15,1.99,-2.6,0.23,-3.4,-3.31
D15__MF24,il4ra2026,MF,tumor,advanced,0.247,643,1958,1.79,2.19,1.54,1.47,-0.168,1.12,0.107,0.314,0.35,0.636,-0.387,0.963,0.723,0.548,1.6,0.33,0.509,0.721,0.553,-0.853,0.565,0.518,0.531,0.908,-0.349,2.89,-1.77,-3.67,-0.867,1.55,1.92,-0.519,-2.7
D11__P4,brentuximab2026,MF,plaque,advanced,0.254,174,512,1.09,1.08,2.57,2.33,0.943,1.02,1.79,0.707,0.459,0.947,-0.427,-0.256,0.609,1.08,1.05,0.522,0.307,0.852,0.724,0.542,1.09,0.327,1.42,1.03,-0.131,3.49,0.446,-2.59,-1.15,0.637,0.457,0.973,0.0864
D15__MF34,il4ra2026,MF,tumor,advanced,0.264,248,692,1.18,1.66,0.537,1.07,-0.201,-0.113,1.31,0.362,0.156,0.302,-0.724,1.57,1.82,0.917,0.606,0.622,0.284,-0.276,1.65,2.27,0.244,1.39,0.0984,-1.7,2.96,-0.464,-0.159,-0.96,-2.28,-2.11,0.177,-1.66,-2.32
Li2024_atlas__CTCL6,li2024,MF,mixed,early,0.277,3937,10298,1.18,5.31,0.83,0.377,1.67,1.64,0.492,2.02,0.983,-0.406,1.09,2.36,0.632,1.31,-1.82,0.599,0.135,1.5,0.711,-1.11,-1.19,0.0399,-0.00484,-3.47,1.32,0,-0.437,-3.38,-1.88,-1.85,-0.658,-0.994,0.827
Li2024_atlas__CTCL14,li2024,MF,skin_unspecified,early,0.288,63,156,1.45,0.506,0.356,-0.333,0.00661,0.0267,0.463,-0.279,0.291,-0.713,1.83,0.663,0.808,0.0883,-0.344,1.05,0.651,0.521,0.0932,-2.3,-0.792,0.269,1.58,-0.803,2.03,0,-0.203,-0.262,0.583,0.882,-0.462,0.822,0.0469
D7__P73,rindler2021_mc,MF,plaque,advanced,0.327,189,389,0.511,2.4,-0.0144,1.17,0.988,1.03,1.33,1.1,0.492,1.39,0.765,1.2,1.28,1.13,1.17,0.894,1.12,0.931,1.72,1.68,1.8,1.1,0.739,-0.0453,-0.0596,-0.418,0.392,-3.33,0.675,-2.27,-0.481,-2.8,-2.29
D7__P84,rindler2021_mc,MF,plaque,advanced,0.327,1892,3889,0.11,-0.159,3.08,1.16,0.794,1.57,1.42,0.877,0.956,1.37,0.444,-1.16,1.06,1.43,0.8,0.809,0.587,0.559,1.24,0.855,0.895,0.67,0.725,0.934,0.702,-0.366,-0.0133,-1.92,-0.796,0.239,0.0568,-1.45,0.342
H__SS1,herrera2021,SS,patch,advanced,0.33,36,73,1.1,-2.33,0.849,0.447,-0.12,0.235,0.903,-0.233,0.591,0.812,-1.48,-2.7,-0.173,0.23,1.06,0.157,0.543,0.0841,-0.488,0.674,0.428,-0.947,-0.39,0.825,-0.84,0.394,-0.733,0.195,-1.12,0.0866,-0.329,0.433,-2.31
Li2024_atlas__PT56,li2024,MF,mixed,advanced,0.343,1292,2478,0.82,0.297,0.411,1.7,0.416,0.822,1.22,0.563,0.625,0.26,-1.06,1.3,-0.185,0.0535,0.831,0.118,0.212,0.189,0.71,0.612,-0.922,-0.00751,0.592,-1.13,-0.245,0,-2.34,-3.45,0.381,0.801,0.789,-2.97,0.408
Li2024_atlas__MF22,li2024,MF,plaque,early,0.343,300,575,2.76,1.52,2.21,2.09,0.79,2.57,1.79,1.87,0.763,0.696,1.9,1.06,0.484,1.59,0.687,0.284,1.25,1.21,0.536,1.8,-0.429,0.249,0.422,-0.735,-0.193,0,-0.73,-4.6,-2.22,-0.439,-0.0422,-3.29,0.372
D13__MF25,gaydosik2022,unresolved,erythroderma,advanced,0.354,99,181,0.372,-1.42,0.803,0.771,-1.11,-0.148,0.192,0.694,0.344,0.0889,0.856,-0.974,-0.0319,0.0851,0.551,0.744,-0.367,0.157,0.398,0.662,-0.367,-0.173,-0.018,0.7,1.41,-0.858,1.63,-0.468,0.34,0.101,0.139,1.27,-0.515
D1__P204,chennareddy2025,MF,tumor,advanced,0.375,3444,5741,0.831,1.2,1.96,1.3,1.4,0.457,0.823,0.968,1.15,0.675,1.17,0.384,0.968,0.934,1.22,0.678,0.761,0.777,0.921,1.35,0.728,0.901,0.558,1.45,0.49,2.84,-0.347,-1.78,-1.03,1.09,0.733,0.954,0.0856
Li2024_atlas__CTCL2,li2024,MF,mixed,early,0.382,1357,2200,3.33,3.51,0.903,0.308,0.142,1.19,1.22,1.44,0.483,1.6,2.45,1.86,1.58,1.33,0.622,1.04,1.31,1.66,1.15,1.14,0.613,0.6,1.01,-2.28,0.221,0,-0.237,-4.16,1.74,-3.25,0.735,-3.3,0.689
Li2024_atlas__PT50,li2024,MF,mixed,early,0.441,1464,1858,0.504,1.34,1.43,0.0946,0.639,0.365,0.0975,0.0712,1.34,-1.58,-0.851,-1.3,0.0772,0.0516,-1.31,0.257,0.75,0.65,0.868,-0.103,0.893,0.752,-0.3,-0.631,0.945,0,-0.893,-3.95,-1.17,-1.06,-2.01,-1.41,-2.14
D13__MF18,gaydosik2022,MF,tumor,advanced,0.469,826,935,-0.208,2.82,-1.89,-0.759,1.12,0.821,-0.812,1.12,2.06,-2.98,0.295,1.23,-0.99,0.158,-1.87,1.39,0.276,0.207,0.873,-0.184,0.404,0.401,-0.0568,-0.772,-0.617,-0.816,0.759,0.707,1.92,-3.32,-3.41,-3.01,-1.28
D6__MF2,gaydosik2019,SS,erythroderma,advanced,0.489,677,708,0.0463,-1.9,2.76,1.51,-0.113,0.515,0.371,0.323,0.566,2.01,-1.07,-1.39,0.359,0.37,0.304,0.205,0.22,0.548,0.42,1.2,0.742,0.0211,0.0595,1.26,0.251,1.27,-0.706,-3.21,-0.133,-0.132,-0.626,0.789,-0.183
D8__P65,alkon2024,MF,patch,early,0.496,615,626,2.95,1.31,-0.0683,1.26,1.16,0.901,1.12,1.17,1.34,1.77,2.51,1.54,1.26,1.06,1.05,1.31,1.19,0.826,0.901,0.749,1.44,0.665,1.38,0.314,0.858,-0.27,0.308,-2.34,-0.466,-0.823,0.728,-2.51,0.771
D8__P138,alkon2024,MF,patch,early,0.505,627,614,3.11,2.4,1.08,1.35,0.96,0.379,0.309,0.543,0.0499,-0.627,1.61,1.1,0.484,0.0202,-0.974,-1.19,-0.214,0.793,0.447,-0.924,-0.691,0.61,0.849,-1.7,1.19,-0.535,0.476,-2.69,0.787,-0.0584,1.1,-2.62,0.813
D13__MF19,gaydosik2022,unresolved,tumor,advanced,0.518,253,235,1.25,-0.487,1.32,0.188,-0.265,0.834,0.694,0.75,-0.0314,1.77,0.83,-0.703,0.645,0.445,1.03,0.683,1.06,0.499,0.0817,0.24,1,0.217,0.168,1.69,0.147,2.76,-0.45,-2.44,-0.662,0.403,0.215,0.311,0.0576
Li2024_atlas__MF30,li2024,MF,tumor,advanced,0.549,2342,1923,2.32,2.54,1.69,2.33,1.05,1.73,1.43,1.28,1.22,0.849,1.9,1.18,1.1,0.886,1.34,0.585,1.04,1.26,1.03,1.1,1.54,0.565,0.51,2.59,1.71,0,-0.326,-4.53,-2.38,0.754,0.48,0.744,-0.933
D1__P312,chennareddy2025,MF,plaque,advanced,0.566,717,550,2.64,3.36,2.93,1.51,1.33,1.09,0.692,1.31,1.32,0.599,0.307,1.26,1.66,-0.346,0.654,0.249,0.45,0.682,1.24,0.968,2.48,1,0.698,-1.07,1.06,3.54,-1.54,-3.8,-1.16,0.215,0.513,0.567,0.0869
D6__SC157,gaydosik2019,MF,tumor,advanced,0.572,791,593,0.342,2.47,-0.512,-0.0761,1.04,0.0924,0.691,0.423,0.209,-1.94,1.89,1.99,0.00142,-0.574,-0.388,0.626,0.527,0.367,0.804,-1.98,-0.741,0.648,0.902,-1.03,-0.209,3.26,0.801,0.898,1.02,-1.68,-1.59,-2.65,-3.64
D15__MF30,il4ra2026,MF,erythroderma,advanced,0.584,497,354,0.459,1.28,-0.303,-0.322,1.19,0.361,-0.0424,0.289,0.308,-0.336,1.08,1.22,0.143,0.183,-0.147,-0.164,0.172,0.442,0.417,-0.465,1,0.433,0.249,-0.45,1.61,-3.08,0.867,-0.176,0.0219,-0.144,0.213,-2.22,0.268
H__MFIV1,herrera2021,MF,plaque,advanced,0.62,44,27,1.91,1,2.44,0.46,0.0948,1.05,1.62,1.76,1.69,2.09,0.877,-1.74,0.582,2.12,1.43,0.394,1.31,-0.938,-0.119,0.335,0.101,-0.586,1.08,0.915,-0.189,2.23,-0.73,-3.88,-1.46,0.467,0.0994,-0.359,0.077
Li2024_atlas__PT52,li2024,MF,mixed,advanced,0.643,160,89,-0.849,0.454,0.159,3.48,0.473,0.413,1.3,1.37,0.757,0.313,0.0754,-0.0776,0.376,1.69,0.248,1.34,0.671,0.908,0.472,1.41,-1.1,0.537,0.777,1.63,0.662,0,-1.65,-2.68,-0.757,0.696,0.644,1.86,-0.248
Li2024_atlas__CTCL5,li2024,MF,mixed,advanced,0.649,7532,4073,0.887,3.22,1.02,0.811,1.56,1.32,1.09,0.858,1.02,0.438,1.88,1.63,1.36,0.704,0.613,1.17,0.989,1.2,0.494,0.245,0.114,-0.297,0.686,-2.04,-0.13,0,-0.845,-0.359,-1.39,-3.65,1.15,-3.53,-4.47
Li2024_atlas__PT47,li2024,MF,mixed,advanced,0.65,307,165,1.39,2.17,2.67,2.28,1.15,1.27,0.909,0.65,1.32,0.898,0.212,1.53,0.675,0.69,0.181,0.75,0.979,0.855,0.81,1.29,0.749,0.283,0.625,1.18,-0.403,0,0.133,-3.21,-1.14,-0.403,0.274,-0.0183,-0.724
Li2024_atlas__PT53,li2024,MF,mixed,advanced,0.658,1417,735,-0.357,-0.121,-0.181,1.56,2.16,0.286,0.547,1.19,1.64,0.562,2.76,2.17,0.649,0.617,0.171,0.911,1.11,0.734,0.968,-1.56,0.172,0.759,0.308,0.0732,1.75,0,1.17,0.0971,1.69,-2.15,-2.13,-2.38,-0.862
B5__PT09,buus2025,SS,erythroderma,,0.664,10427,5273,1.92,0.471,0.727,0.887,-0.837,0.433,1.1,0.576,0.734,0.502,0.846,-0.459,0.36,1.26,0.169,0.122,0.753,-0.474,0.0414,1.2,0.177,-0.8,0.741,0.2,0.659,2.63,-0.0682,-3.3,-0.841,-0.736,-1.18,-0.328,0.15
Li2024_atlas__CTCL13,li2024,MF,skin_unspecified,advanced,0.674,5962,2884,1.22,2.1,1.84,0.957,0.737,1.59,0.597,1.14,0.698,0.786,0.0477,-1.34,0.734,0.426,-0.169,0.636,0.483,0.781,0.196,0.929,1.72,0.499,-0.00515,0.9,-0.415,0,0.218,-3.6,-1.21,0.798,0.181,0.35,0.284
D8__P163,alkon2024,MF,patch,early,0.679,881,417,1.94,-0.853,1.53,0.706,1.55,0.282,1.14,0.84,0.803,0.869,0.85,1.62,0.885,0.309,0.522,0.244,0.926,1.64,0.588,0.437,-1.56,0.616,1.31,-1.26,0.69,-0.149,-1.47,-3.04,-0.402,-1.25,-0.56,-1.91,0.748
Li2024_atlas__CTCL17,li2024,MF,skin_unspecified,advanced,0.712,203,82,0.098,0.429,-1.17,-1.26,0.525,-0.391,0.0376,0.163,0.401,-0.531,0.126,-0.176,-0.132,0.377,0.351,-0.401,0.427,-0.18,0.256,-1.51,0.163,0.351,-0.0232,-1.73,0.272,0,0.376,-0.519,-0.173,-1.44,-0.645,-1.14,-0.321
Li2024_atlas__MF21,li2024,MF,tumor,advanced,0.713,1761,708,1.09,3.66,2.97,2.98,1.03,0.963,1,0.532,1.44,1.4,-0.761,-0.773,1.34,0.81,-0.278,0.435,0.601,0.261,0.889,1.27,2.18,0.763,-0.0845,1.25,-0.385,0,0.133,-5.09,-2.13,0.265,1.05,1.72,-0.125
D8__P150,alkon2024,MF,patch,early,0.724,2393,912,0.0924,0.237,-0.439,0.108,1.18,0.953,1.32,0.375,0.392,-0.566,-0.0893,1.8,0.34,0.0104,0.301,0.343,-0.0606,0.873,0.463,0.675,0.931,0.595,0.835,-0.0474,-0.0928,0,0.764,-2.7,1.79,0.591,0.589,-2.77,0.176
D7__MF311,rindler2021_mc,MF,patch,advanced,0.748,5693,1919,0.412,-0.692,1.85,0.538,0.0163,0.375,0.815,0.031,0.137,1.2,-0.567,-2.6,0.632,0.43,0.604,1.18,0.127,-0.0273,0.273,1.2,-0.492,0.493,0.206,1.15,-0.0016,2.61,-1.59,-3.53,-1.15,0.0217,-0.678,0.994,-0.626
Li2024_atlas__CTCL8,li2024,MF,mixed,early,0.751,7379,2447,-0.593,0.333,-0.0487,0.322,0.737,0.381,0.731,1.91,0.336,0.916,0.284,2,0.779,-0.514,1.48,0.615,0.711,0.493,0.877,1.62,-0.373,0.166,0.884,-1.04,0.0487,0,-1.09,-3.4,-0.114,-1.62,-1.35,-1.98,0.0092
D7__MF309,rindler2021_mc,MF,patch,advanced,0.755,5395,1748,2.52,3.44,-1.79,0.588,1.75,0.148,0.492,0.764,1.39,-0.838,0.16,1.98,-0.489,0.503,-0.488,0.378,0.584,0.748,0.241,-1.27,2.07,0.0381,-0.394,-0.327,0.893,-0.494,0.441,-1.31,1.58,-1.88,-0.23,0.559,-0.609
D6__SC158,gaydosik2019,MF,plaque,advanced,0.756,306,99,-0.124,-0.756,1.88,2.06,-0.135,0.243,0.726,0.651,0.415,1.2,-0.559,-0.4,-0.584,1.33,0.506,-0.584,-0.0428,-0.285,-0.204,0.327,0.276,-0.494,0.0633,0.952,0.302,1.93,-0.305,-2.53,0.449,0.304,0.133,-0.604,-1.68
D11__P6,brentuximab2026,MF,plaque,advanced,0.769,4575,1376,0.812,-0.395,-0.42,0.305,1.69,1.85,1.16,1.14,1.47,1.68,2.83,1.34,1.06,1.77,1.63,1.98,1.48,1.98,0.966,0.759,0.214,0.826,0.508,1.92,0.74,2.8,-1.84,-3.99,-1.37,-2.25,0.385,-0.364,-0.377
Li2024_atlas__CTCL16,li2024,MF,skin_unspecified,early,0.779,1037,294,1.53,1.24,1.16,1.76,1.32,0.6,0.393,0.0784,1.1,0.292,0.865,0.313,0.885,0.125,-0.384,0.29,0.0836,0.158,0.584,0.611,0.977,0.755,0.914,-0.508,0.603,0,0.0801,-0.215,-0.351,0.177,0.0957,0.189,0.224
D15__MF35,il4ra2026,MF,mixed,advanced,0.786,2187,597,1.01,2.15,-0.583,0.405,0.808,1.27,0.377,1.24,1.19,0.549,0.73,1.06,1.07,0.459,0.939,0.901,1.18,1.1,0.795,1.62,-0.113,0.545,-0.00837,0.907,0.56,-0.332,1.22,-2.84,-1.85,-0.299,-0.0256,-2.33,0.0715
D3__P196,brunner2024,MF,erythroderma,advanced,0.791,506,134,2.39,0.19,3.55,0.62,-0.264,0.214,0.548,0.107,-0.0505,0.619,0.758,-0.238,0.446,0.275,1.53,0.077,0.439,0.286,0.328,0.654,0.698,0.399,0.594,2.43,1.49,0,0.592,-1.31,-2.18,0.284,0.0182,1.83,0.143
D3__P192,brunner2024,SS,erythroderma,advanced,0.813,2874,660,-0.00183,0.24,2.59,-0.125,-0.484,0.417,0.692,0.393,-0.101,0.9,0.25,-0.685,0.387,-0.361,1.27,0.903,-0.0119,-0.127,0.468,1.63,-0.0016,0.583,-0.481,2.18,0.132,1.39,0.2,-3.52,-0.971,1.04,-0.0972,0.73,-0.316
H__SS4,herrera2021,SS,erythroderma,advanced,0.816,204,46,0.892,0.783,1.14,0.796,0.038,0.842,0.631,1.11,1.12,1.16,0.32,-0.232,1.19,1.18,1.1,1.14,1.82,0.226,1.04,1.44,0.681,0.23,0.624,2.2,0.328,2.26,-0.161,-3.78,-2.26,1.32,0.864,1.13,-1.45
D1__PGS,chennareddy2025,MF,erythroderma,advanced,0.817,2056,459,1.06,1.4,1.56,0.729,-0.372,0.861,1.38,0.914,0.545,1.98,0.69,0.119,0.356,0.628,1.25,0.899,0.441,0.525,0.449,0.374,0.931,0.588,-0.0762,0.707,-0.0485,2.28,-0.0789,-3.97,-0.874,0.99,0.282,0.311,-0.313
D11__P3,brentuximab2026,MF,tumor,advanced,0.827,7596,1589,1.49,1.69,1.72,1.24,1.3,1.11,1.28,0.725,1.04,1.3,0.919,-0.297,1.16,0.784,1.16,0.817,0.526,0.572,0.928,1.02,1.52,0.582,0.499,1.02,0.484,2.99,0.223,-4.53,-1.69,-1.9,0.985,1.6,0.223
D11__P5,brentuximab2026,MF,tumor,advanced,0.827,6550,1370,0.618,1.03,2.63,2.64,1.8,-0.44,0.572,0.744,0.857,1.32,0.524,-0.794,1.32,1.16,0.614,0.66,0.698,0.992,0.389,0.202,-0.486,0.92,0.545,1.74,0.332,2.56,-0.487,-4.44,-2.23,0.191,0.435,1,0.179
Li2024_atlas__CTCL3,li2024,MF,mixed,early,0.832,8823,1778,0.23,2.58,3.11,-0.0368,1.24,1.5,0.421,0.781,1.07,1.46,2.45,2.27,-0.44,0.883,0.835,0.647,1.01,0.661,0.797,-3.68,0.681,0.123,0.818,-1.02,0.0657,0,-0.773,-5.34,2.85,-3.91,0.215,-3.73,-4.52
B5__PT23,buus2025,SS,erythroderma,,0.836,6371,1247,2.07,-0.629,2.57,0.949,-0.173,0.933,0.416,0.11,-0.0351,1.8,-0.765,-0.0758,-0.619,1.23,0.845,0.107,0.0713,-0.243,0.147,-0.944,0.276,-0.23,-0.043,1.31,0.582,-2,0.875,0.8,-1.12,-0.213,-0.0874,1.51,-0.539
D15__MF48,il4ra2026,MF,papule,advanced,0.837,3650,709,1.14,0.944,0.464,0.598,1.09,1.98,0.558,1.07,0.114,0.856,0.947,2.86,1.13,0.00688,0.627,0.117,0.345,0.594,0.845,0.7,1.07,0.936,1.58,-0.671,1.67,-0.0494,-1.55,-0.745,0.271,-0.662,-1.04,-2.39,0.337
Li2024_atlas__MF28,li2024,MF,tumor,advanced,0.845,10395,1904,0.78,1.07,0.662,1.36,1.08,0.0337,0.887,0.345,1.38,0.698,0.218,1.01,0.713,-0.194,0.844,-0.0764,0.271,0.286,0.598,0.0802,-0.0732,0.463,-0.0518,0.324,0.527,0,-1.53,-4.1,-1.97,-0.32,0.0614,0.382,-0.965
Li2024_atlas__PT11,li2024,MF,patch,advanced,0.874,1948,282,0.0567,3.58,-0.103,2.31,1.45,1.21,0.962,2.43,2.09,-0.797,-0.073,-1.24,-0.408,1.73,-1.67,0.793,1.69,1.11,0.749,0.399,-0.695,1.17,0.694,-0.918,0.0315,0,-0.855,0.76,-0.121,1.36,1.01,0.799,-3.57
Li2024_atlas__CTCL10,li2024,MF,skin_unspecified,early,0.88,1151,157,1.18,-1.18,-0.536,1.46,1.35,1.19,-0.187,1.16,1.16,-0.753,0.795,1.23,-0.642,1.05,0.784,1.12,1.16,0.639,0.961,0.934,-1.1,0.507,0.607,-3.3,1.1,0,-0.674,-3.76,-1.4,-3.3,-3.77,-3.74,-0.0226
D11__P1,brentuximab2026,MF,plaque,advanced,0.883,678,90,0.303,-0.013,2.06,2.97,1.29,1.13,1.01,0.479,0.351,1.03,-1.86,-1.37,0.783,0.541,1.15,0.0421,0.766,0.876,0.382,0.0259,0.633,-0.477,0.398,2.07,-0.424,3.49,-2.57,-4.48,-1.44,1.64,1.41,1.61,-0.0295
Li2024_atlas__CTCL12,li2024,MF,skin_unspecified,advanced,0.894,2784,331,1.4,-0.373,2.64,1.74,0.11,0.884,1.03,0.774,0.359,0.706,0.42,0.409,0.606,0.124,1.59,1.27,0.781,0.746,0.984,1.83,0.213,1.55,0.309,1.84,2.43,0,1.75,-3.33,-1.08,0.668,-0.471,-3.29,0.161
D11__P2,brentuximab2026,MF,tumor,advanced,0.899,9473,1063,2.37,2.22,1.46,1.49,1.35,0.231,1.51,0.242,1,0.747,1.07,1.62,1.46,0.0148,0.546,0.518,0.657,0.291,0.925,1.08,2,0.791,0.406,0.162,1.16,2.6,-0.495,-4.18,-1.67,0.111,0.0919,0.331,0.138
D1__P303,chennareddy2025,MF,tumor,advanced,0.899,1038,116,0.105,1.33,0.951,0.398,1.45,0.778,2.13,0.961,1.11,2.37,1.8,1.69,2.03,0.403,1.71,1.75,1,0.978,1.08,1.34,0.552,0.911,0.38,0.563,0.732,3.26,-0.0586,-5.04,-1.19,-2.5,-2.87,-3.1,0.28
Li2024_atlas__PT35,li2024,MF,mixed,advanced,0.919,3502,308,2.45,2.85,2.15,0.485,0.0332,1.53,2.93,0.32,0.437,2.21,0.623,1.36,1.6,0.95,2.65,2.05,0.997,0.706,0.579,-2.03,0.376,0.764,0.533,2.95,0.477,0,1.84,-5.72,0,2.89,2.4,1.64,1.88
D15__MF26,il4ra2026,MF,erythroderma,advanced,0.92,3041,265,1.24,-1.06,-0.474,0.994,0.524,0.456,0.99,0.491,0.31,1.21,-1.11,-2.78,-0.742,0.0477,1.98,1.31,0.461,0.19,-0.0434,-0.00902,0.127,-0.1,0.226,0.667,-0.462,3.41,-0.833,-0.613,-1.35,0.781,0.426,0.929,-0.305
D1__P171,chennareddy2025,MF,tumor,advanced,0.932,5216,382,1.99,0.378,1.82,1.45,-0.152,1.2,0.813,0.601,0.247,2.04,0.326,0.185,0.717,1.05,1.9,1.51,0.444,0.248,0.111,0.527,1.23,0.368,0.549,1.95,0.0827,1.49,0.519,-4.09,-1.01,1.88,1.52,0.926,-1.02
Li2024_atlas__CTCL9,li2024,MF,skin_unspecified,early,0.954,2216,106,2.15,-0.027,0.282,0.966,0.215,0.973,0.978,0.684,1.28,-0.195,1.78,1.13,1.09,0.24,-0.0849,-0.128,1.06,-0.509,0.446,1.39,0.901,0.798,0.21,0.511,2.05,0,1.01,-2.26,-0.39,-3.28,-1.13,-1.94,-0.306
Li2024_atlas__CTCL1,li2024,MF,mixed,advanced,0.956,7676,352,0.39,2.95,0.762,0.083,-0.276,0.521,0.167,0.168,0.835,0.749,-1.65,-2.24,0.375,0.732,0.649,1.04,0.449,0.242,0.0693,1.15,1.99,0.0582,0.134,0.689,-0.753,0,-0.451,-5.25,-1.29,0.334,-0.215,0.383,-0.821
Li2024_atlas__PT55,li2024,MF,mixed,advanced,0.968,2177,71,0.211,3.46,-1.65,1.23,2.61,-0.0261,-0.712,1.61,1.18,-3.51,0.647,2.38,-0.785,1.47,-1.45,0.591,1.33,0.706,0.5,-1.48,1.74,0.347,-0.063,-2.3,-0.0181,0,-1.46,-0.287,0.875,-1.06,-3.54,-3.69,-4.8
Li2024_atlas__CTCL18,li2024,MF,skin_unspecified,advanced,0.982,7004,127,1.14,0.744,-0.589,-0.755,0.657,0.721,0.735,0.442,0.819,-1.59,0.911,-0.334,-0.161,-0.0667,-0.901,0.892,0.737,0.349,0.967,-2.14,-0.0811,0.571,0.591,-0.96,-0.161,0,-0.394,0.377,1.03,-3.85,-1.9,-2.02,0.216
D13__MF17,gaydosik2022,MF,tumor,advanced,0.985,3696,56,1.12,-0.465,-0.797,0.781,-0.0206,0.731,0.371,0.392,-0.762,0.823,-2.1,-3.76,-1.9,0.806,1.08,-0.0545,-0.484,0.129,0.204,-0.877,1.4,-0.334,-0.118,0.41,-0.887,-2.42,-2.17,-5.41,-1.35,0.796,0.55,0.624,-0.792
```

## §6b Per-donor detection, malignant CD4

**What:** share of each patient's malignant CD4 cells with >= 1 UMI of the gene.

**How:** pct_expr of the CD4_malignant arm; paired donors, burden order.

**Full table on disk:** `tables/tgt49_donor_pct_malignant.csv`

```csv
donor,study,dx2,lesion,clin_stage_class,burden,n_malignant,n_reactive,CD70,CD9,CPM,COL6A3,CD151,GRN,TFRC,SLC39A6,SLC4A2,TNFRSF25,SLC1A4,LGALS3,CD82,SLC5A3,IL6R,SYPL1,CALU,QSOX1,VDAC1,TNFRSF4,TSPAN2,GPI,TMEM30A,CCR4,TNFRSF8,KIR3DL2,IL2RA,CD7,DPP4,CD5,CD6,CD27,CD2
Li2024_atlas__MF17,li2024,MF,tumor,advanced,0.00837,37,4385,0.0811,0,0.0541,0.027,0.297,0.27,0.297,0.243,0.216,0.216,0.189,0.73,0.703,0.108,0.459,0.486,0.27,0.351,0.757,0.027,0.189,0.757,0.432,0.027,0.108,0,0.0541,0,0.459,0.649,0.73,0.0541,0.757
Li2024_atlas__CTCL7,li2024,MF,mixed,early,0.0869,3658,38432,0.171,0.671,0.0216,0.00711,0.201,0.12,0.0293,0.469,0.0746,0.0243,0.603,0.854,0.524,0.154,0.0123,0.227,0.105,0.109,0.381,0.233,0.015,0.55,0.186,0.0213,0.0161,0,0.0897,0.0596,0.0585,0.261,0.223,0.0528,0.809
Li2024_atlas__MF27,li2024,MF,tumor,advanced,0.129,107,725,0.729,0.0561,0.168,0.14,0.43,0.393,0.383,0.318,0.271,0.121,0.271,0.664,0.748,0.346,0.15,0.43,0.29,0.393,0.71,0.43,0.196,0.738,0.383,0.234,0.196,0,0.308,0.364,0.00935,0.645,0.561,0.645,0.888
Li2024_atlas__CTCL4,li2024,MF,mixed,early,0.136,1241,7906,0.32,0.468,0.0338,0.00645,0.142,0.176,0.124,0.415,0.0669,0.138,0.427,0.73,0.326,0.252,0.0572,0.346,0.169,0.0894,0.367,0.118,0.05,0.611,0.168,0.0927,0.058,0,0.0927,0.218,0.0274,0.151,0.396,0.0677,0.595
D15__MF28,il4ra2026,MF,tumor,advanced,0.151,123,691,0.293,0.325,0.439,0.0407,0.0488,0.106,0.374,0.187,0.0732,0.0732,0.26,0.407,0.122,0.634,0.268,0.317,0.358,0.203,0.341,0.195,0.0325,0.293,0.382,0.293,0.0325,0,0.439,0.0976,0.0894,0.0976,0.106,0.0244,0.407
D10__JONAK1,jonak2021,MF,plaque,early,0.153,348,1920,0.0431,0.161,0.103,0.195,0.273,0.201,0.21,0.135,0.181,0.328,0.0287,0.106,0.684,0.0718,0.342,0.428,0.164,0.0718,0.46,0.477,0.138,0.491,0.175,0.29,0.0172,0.172,0.0776,0.0517,0.00575,0.583,0.503,0.514,0.572
D13__MF21,gaydosik2022,MF,tumor,advanced,0.177,118,547,0.432,0.11,0.5,0.0763,0.424,0.373,0.441,0.364,0.449,0.229,0.39,0.314,0.576,0.153,0.0508,0.534,0.381,0.331,0.585,0.136,0.127,0.602,0.169,0.475,0,0.432,0.508,0.0763,0.0339,0.415,0.195,0.695,0.602
D3__P105,brunner2024,MF,erythroderma,advanced,0.185,117,516,0.0855,0.0769,0.154,0.0598,0.0171,0.12,0.0684,0.0684,0.0342,0.12,0.0171,0.145,0.35,0.12,0.248,0.222,0.0256,0.0171,0.188,0.239,0.162,0.0684,0.239,0.573,0,0.0769,0.0598,0.00855,0.00855,0.111,0.137,0.179,0.274
D15__MF36,il4ra2026,MF,mixed,advanced,0.221,620,2182,0.148,0.497,0.431,0.221,0.223,0.206,0.668,0.398,0.113,0.469,0.281,0.427,0.495,0.615,0.285,0.505,0.287,0.235,0.615,0.465,0.181,0.629,0.274,0.331,0.105,0.513,0.263,0.0935,0.00806,0.476,0.702,0.237,0.735
D8__P90,alkon2024,MF,patch,early,0.227,68,232,0.103,0.324,0,0.103,0.103,0.0588,0.147,0.118,0.0147,0.0294,0.191,0.647,0.25,0.0882,0,0.235,0.0882,0.0147,0.338,0,0.103,0.397,0.0588,0,0.0588,0,0.206,0.0588,0.176,0.0147,0.221,0,0.0588
D15__MF24,il4ra2026,MF,tumor,advanced,0.247,643,1958,0.255,0.201,0.263,0.238,0.0762,0.24,0.303,0.151,0.0669,0.372,0.0529,0.286,0.398,0.316,0.393,0.218,0.146,0.177,0.314,0.201,0.0778,0.4,0.114,0.174,0.014,0.347,0.124,0.0389,0,0.644,0.731,0.269,0.0731
D11__P4,brentuximab2026,MF,plaque,advanced,0.254,174,512,0.391,0.569,0.805,0.557,0.397,0.443,0.77,0.282,0.161,0.569,0.184,0.707,0.707,0.385,0.379,0.489,0.253,0.253,0.621,0.575,0.322,0.54,0.379,0.46,0.138,0.678,0.391,0.0287,0.0172,0.678,0.718,0.626,0.805
D15__MF34,il4ra2026,MF,tumor,advanced,0.264,248,692,0.222,0.177,0.0444,0.0645,0.0806,0.0927,0.476,0.202,0.0484,0.282,0.0927,0.48,0.52,0.548,0.266,0.403,0.153,0.101,0.673,0.746,0.0484,0.722,0.177,0.0242,0.254,0,0.254,0.423,0.0121,0.0282,0.508,0.00403,0.0806
Li2024_atlas__CTCL6,li2024,MF,mixed,early,0.277,3937,10298,0.115,0.878,0.105,0.0168,0.343,0.386,0.156,0.501,0.143,0.189,0.274,0.925,0.407,0.403,0.0193,0.487,0.226,0.252,0.645,0.195,0.00381,0.694,0.228,0.0318,0.0704,0,0.0813,0.0533,0.00102,0.187,0.384,0.151,0.799
Li2024_atlas__CTCL14,li2024,MF,skin_unspecified,early,0.288,63,156,0.143,0.286,0.0476,0.0317,0.159,0.159,0.333,0.127,0.19,0.111,0.143,0.365,0.556,0.333,0.0952,0.0476,0.159,0.0952,0.127,0,0.238,0.0476,0.111,0.0952,0.0635,0,0.143,0.0635,0.0476,0.365,0.111,0.222,0.286
D7__P73,rindler2021_mc,MF,plaque,advanced,0.327,189,389,0.0899,0.36,0.00529,0.0847,0.222,0.212,0.185,0.19,0.0317,0.323,0.0529,0.683,0.439,0.153,0.164,0.466,0.228,0.0476,0.661,0.407,0.286,0.571,0.233,0.138,0.00529,0,0.159,0.037,0.122,0.0106,0.153,0.0317,0.106
D7__P84,rindler2021_mc,MF,plaque,advanced,0.327,1892,3889,0.108,0.0624,0.444,0.138,0.172,0.355,0.289,0.212,0.0941,0.205,0.056,0.154,0.628,0.208,0.234,0.452,0.161,0.0555,0.538,0.277,0.122,0.463,0.172,0.29,0.0423,0.00106,0.124,0.0909,0.00476,0.395,0.321,0.0946,0.724
H__SS1,herrera2021,SS,patch,advanced,0.33,36,73,0.361,0.0278,0.0833,0.278,0.111,0.25,0.417,0.139,0.111,0.472,0.0556,0.111,0.583,0.25,0.528,0.444,0.167,0.139,0.333,0.778,0.306,0.222,0.194,0.5,0.0278,0.0278,0.0556,0.472,0,0.361,0.361,0.639,0.111
Li2024_atlas__PT56,li2024,MF,mixed,advanced,0.343,1292,2478,0.121,0.0689,0.0294,0.134,0.133,0.319,0.269,0.0712,0.0666,0.205,0.00387,0.717,0.442,0.0488,0.257,0.211,0.0937,0.0751,0.51,0.445,0.031,0.384,0.166,0.041,0.0387,0,0.0163,0.0341,0.0751,0.569,0.622,0.0325,0.831
Li2024_atlas__MF22,li2024,MF,plaque,early,0.343,300,575,0.517,0.127,0.227,0.22,0.29,0.66,0.61,0.527,0.277,0.693,0.43,0.79,0.637,0.43,0.517,0.597,0.307,0.477,0.693,0.953,0.12,0.677,0.44,0.16,0.0367,0,0.13,0.0267,0.00333,0.503,0.68,0.0133,0.933
D13__MF25,gaydosik2022,unresolved,erythroderma,advanced,0.354,99,181,0.0909,0.0202,0.111,0.202,0.0404,0.101,0.283,0.111,0.101,0.232,0.141,0.313,0.354,0.172,0.192,0.202,0.101,0.0909,0.374,0.202,0.0505,0.303,0.0606,0.141,0.0707,0,0.162,0.232,0.0202,0.283,0.414,0.525,0.434
D1__P204,chennareddy2025,MF,tumor,advanced,0.375,3444,5741,0.215,0.262,0.199,0.166,0.326,0.151,0.178,0.196,0.14,0.269,0.143,0.363,0.598,0.166,0.402,0.361,0.173,0.155,0.494,0.398,0.101,0.52,0.287,0.456,0.0267,0.53,0.121,0.181,0.00813,0.492,0.51,0.663,0.662
Li2024_atlas__CTCL2,li2024,MF,mixed,early,0.382,1357,2200,0.35,0.542,0.0656,0.00737,0.13,0.36,0.166,0.355,0.0774,0.644,0.382,0.825,0.517,0.209,0.243,0.68,0.368,0.308,0.786,0.486,0.117,0.815,0.273,0.0818,0.00516,0,0.028,0.0508,0.382,0.014,0.682,0.0103,0.863
Li2024_atlas__PT50,li2024,MF,mixed,early,0.441,1464,1858,0.111,0.161,0.133,0.0526,0.159,0.174,0.293,0.148,0.199,0.0949,0.0376,0.0738,0.411,0.15,0.0485,0.306,0.232,0.149,0.573,0.385,0.235,0.528,0.2,0.111,0.0929,0,0.176,0.028,0.00137,0.197,0.105,0.126,0.142
D13__MF18,gaydosik2022,MF,tumor,advanced,0.469,826,935,0.178,0.439,0.00484,0.0266,0.349,0.331,0.212,0.295,0.483,0.00726,0.108,0.573,0.277,0.262,0.00363,0.603,0.185,0.145,0.637,0.517,0.171,0.57,0.217,0.0291,0,0,0.318,0.7,0.339,0.00484,0.0121,0.0133,0.479
D6__MF2,gaydosik2019,SS,erythroderma,advanced,0.489,677,708,0.0679,0.121,0.353,0.115,0.0473,0.099,0.294,0.14,0.0798,0.417,0.0739,0.313,0.301,0.225,0.105,0.323,0.109,0.0709,0.411,0.453,0.106,0.13,0.0857,0.0739,0.0281,0.202,0.108,0.0148,0.00295,0.109,0.152,0.394,0.511
D8__P65,alkon2024,MF,patch,early,0.496,615,626,0.465,0.556,0.00813,0.148,0.215,0.172,0.205,0.167,0.0699,0.361,0.311,0.854,0.572,0.161,0.221,0.512,0.148,0.0894,0.459,0.39,0.156,0.345,0.315,0.167,0.0374,0.00163,0.156,0.039,0.0146,0.107,0.25,0.0276,0.75
D8__P138,alkon2024,MF,patch,early,0.505,627,614,0.553,0.614,0.0829,0.107,0.164,0.1,0.175,0.0877,0.0303,0.118,0.212,0.718,0.372,0.0797,0.0303,0.0734,0.0686,0.0877,0.359,0.121,0.0303,0.314,0.196,0.0207,0.0829,0,0.185,0.0128,0.0574,0.14,0.368,0.0207,0.735
D13__MF19,gaydosik2022,unresolved,tumor,advanced,0.518,253,235,0.407,0.269,0.217,0.146,0.0711,0.344,0.423,0.17,0.111,0.593,0.221,0.427,0.688,0.304,0.399,0.451,0.249,0.202,0.423,0.431,0.186,0.486,0.17,0.514,0.0791,0.368,0.198,0.126,0.00791,0.506,0.542,0.561,0.538
Li2024_atlas__MF30,li2024,MF,tumor,advanced,0.549,2342,1923,0.336,0.322,0.162,0.211,0.251,0.514,0.298,0.366,0.2,0.576,0.198,0.373,0.702,0.196,0.606,0.676,0.317,0.341,0.673,0.408,0.226,0.697,0.406,0.604,0.122,0,0.0687,0.0243,0.00213,0.788,0.74,0.81,0.505
D1__P312,chennareddy2025,MF,plaque,advanced,0.566,717,550,0.357,0.551,0.303,0.16,0.194,0.192,0.197,0.149,0.099,0.197,0.0795,0.717,0.718,0.0209,0.152,0.238,0.0893,0.0837,0.6,0.563,0.476,0.421,0.145,0.053,0.099,0.597,0.0683,0.0181,0.00279,0.319,0.314,0.53,0.609
D6__SC157,gaydosik2019,MF,tumor,advanced,0.572,791,593,0.204,0.579,0.00759,0.0164,0.324,0.21,0.23,0.188,0.115,0.0139,0.33,0.716,0.226,0.235,0.0177,0.458,0.238,0.13,0.633,0.133,0.00506,0.37,0.224,0.00379,0.00253,0.598,0.303,0.915,0.164,0.119,0.129,0.0215,0.0405
D15__MF30,il4ra2026,MF,erythroderma,advanced,0.584,497,354,0.529,0.515,0.209,0.0966,0.765,0.736,0.702,0.704,0.519,0.608,0.97,0.992,0.94,0.716,0.416,0.674,0.557,0.581,0.928,0.716,0.396,0.968,0.748,0.529,0.535,0.0121,0.952,0.704,0.39,0.809,0.899,0.0503,0.972
H__MFIV1,herrera2021,MF,plaque,advanced,0.62,44,27,0.364,0.227,0.295,0.909,0.295,0.205,0.5,0.364,0.295,0.75,0.205,0.182,0.614,0.636,0.5,0.364,0.409,0.114,0.455,0.591,0.318,0.25,0.25,0.545,0.0682,0.295,0.114,0.0227,0,0.5,0.477,0.25,0.75
Li2024_atlas__PT52,li2024,MF,mixed,advanced,0.643,160,89,0.0563,0.2,0.0375,0.562,0.194,0.2,0.406,0.119,0.194,0.438,0.0375,0.419,0.637,0.356,0.3,0.4,0.2,0.138,0.438,0.581,0.0437,0.475,0.175,0.212,0.0563,0,0.0125,0.0187,0.0125,0.344,0.55,0.537,0.694
Li2024_atlas__CTCL5,li2024,MF,mixed,advanced,0.649,7532,4073,0.145,0.501,0.19,0.0382,0.358,0.421,0.279,0.253,0.108,0.59,0.484,0.876,0.82,0.34,0.282,0.691,0.429,0.19,0.765,0.672,0.0886,0.714,0.256,0.0786,0.00478,0,0.106,0.843,0.00173,0.0178,0.812,0.00624,0.0159
Li2024_atlas__PT47,li2024,MF,mixed,advanced,0.65,307,165,0.235,0.326,0.384,0.436,0.368,0.528,0.459,0.215,0.274,0.635,0.0489,0.831,0.7,0.208,0.306,0.505,0.267,0.218,0.642,0.606,0.202,0.482,0.381,0.41,0.0521,0,0.283,0.0228,0.00326,0.293,0.606,0.508,0.554
Li2024_atlas__PT53,li2024,MF,mixed,advanced,0.658,1417,735,0.00494,0.000706,0.0113,0.157,0.301,0.128,0.325,0.233,0.249,0.466,0.445,0.814,0.603,0.191,0.264,0.518,0.24,0.158,0.594,0.1,0.0741,0.594,0.304,0.172,0.162,0,0.325,0.719,0.303,0.0318,0.0543,0.0346,0.434
B5__PT09,buus2025,SS,erythroderma,,0.664,10427,5273,0.238,0.159,0.248,0.0471,0.0205,0.0943,0.549,0.159,0.155,0.21,0.314,0.395,0.46,0.36,0.29,0.276,0.174,0.107,0.322,0.546,0.0739,0.276,0.246,0.249,0.0781,0.321,0.248,0.049,0.00211,0.247,0.222,0.36,0.644
Li2024_atlas__CTCL13,li2024,MF,skin_unspecified,advanced,0.674,5962,2884,0.271,0.366,0.245,0.491,0.149,0.593,0.736,0.253,0.251,0.802,0.128,0.0844,0.593,0.372,0.186,0.142,0.272,0.298,0.0805,0.426,0.291,0.2,0.0753,0.76,0.0411,0,0.442,0.0335,0.0273,0.665,0.658,0.608,0.699
D8__P163,alkon2024,MF,patch,early,0.679,881,417,0.261,0.0284,0.0953,0.0329,0.166,0.0715,0.142,0.104,0.0295,0.208,0.0499,0.859,0.287,0.0454,0.0885,0.149,0.0681,0.128,0.317,0.156,0.00227,0.257,0.259,0.0125,0.0284,0,0.00341,0.0216,0.0568,0.042,0.114,0.00795,0.692
Li2024_atlas__CTCL17,li2024,MF,skin_unspecified,advanced,0.712,203,82,0.414,0.498,0.00493,0.0197,0.355,0.227,0.33,0.163,0.177,0.271,0.163,0.399,0.217,0.473,0.0148,0.182,0.35,0.0837,0.227,0,0.251,0.222,0.103,0.00985,0.00493,0,0.0739,0.153,0.621,0.00493,0.261,0.00493,0.783
Li2024_atlas__MF21,li2024,MF,tumor,advanced,0.713,1761,708,0.325,0.731,0.546,0.415,0.548,0.401,0.586,0.334,0.425,0.852,0.206,0.274,0.892,0.648,0.227,0.648,0.371,0.326,0.867,0.882,0.675,0.846,0.404,0.791,0.0136,0,0.534,0.0142,0,0.743,0.951,0.923,0.867
D8__P150,alkon2024,MF,patch,early,0.724,2393,912,0.0134,0.0221,0.00167,0.013,0.0978,0.11,0.161,0.0485,0.0209,0.0648,0.00585,0.585,0.301,0.0318,0.183,0.158,0.038,0.0497,0.144,0.194,0.056,0.242,0.161,0.124,0.000418,0,0.129,0.0221,0.218,0.338,0.394,0.0142,0.535
D7__MF311,rindler2021_mc,MF,patch,advanced,0.748,5693,1919,0.0487,0.00422,0.108,0.0283,0.0458,0.0509,0.119,0.0316,0.0102,0.14,0.00386,0.0156,0.257,0.048,0.095,0.221,0.0473,0.0218,0.206,0.195,0.0137,0.246,0.0625,0.108,0.00193,0.23,0.00281,0.0112,0.000703,0.11,0.0736,0.377,0.275
Li2024_atlas__CTCL8,li2024,MF,mixed,early,0.751,7379,2447,0.0279,0.0455,0.00488,0.00881,0.102,0.0213,0.0824,0.236,0.0302,0.137,0.0518,0.909,0.232,0.04,0.101,0.32,0.136,0.0171,0.325,0.462,0.0159,0.172,0.146,0.0855,0.000949,0,0.0281,0.0594,0.0314,0.00718,0.013,0.0218,0.338
D7__MF309,rindler2021_mc,MF,patch,advanced,0.755,5395,1748,0.348,0.551,0.0106,0.0799,0.275,0.0842,0.198,0.121,0.107,0.0241,0.0267,0.689,0.271,0.0832,0.0538,0.246,0.13,0.0649,0.389,0.106,0.236,0.35,0.0643,0.089,0.0638,0.000927,0.192,0.155,0.114,0.0475,0.0938,0.468,0.418
D6__SC158,gaydosik2019,MF,plaque,advanced,0.756,306,99,0.157,0.225,0.203,0.111,0.108,0.127,0.154,0.0817,0.049,0.324,0.0752,0.389,0.229,0.307,0.098,0.154,0.0784,0.0621,0.34,0.314,0.0425,0.141,0.105,0.0425,0.0392,0.098,0.176,0.137,0.0131,0.199,0.235,0.317,0.229
D11__P6,brentuximab2026,MF,plaque,advanced,0.769,4575,1376,0.261,0.109,0.0111,0.133,0.538,0.602,0.479,0.308,0.265,0.591,0.706,0.792,0.609,0.406,0.484,0.726,0.464,0.489,0.633,0.297,0.108,0.651,0.285,0.704,0.0544,0.544,0.00634,0.023,0.0024,0.0361,0.612,0.329,0.6
Li2024_atlas__CTCL16,li2024,MF,skin_unspecified,early,0.779,1037,294,0.195,0.238,0.455,0.571,0.2,0.251,0.561,0.122,0.237,0.581,0.166,0.234,0.575,0.319,0.162,0.0694,0.165,0.152,0.254,0.392,0.148,0.205,0.0521,0.429,0.103,0,0.304,0.314,0.0135,0.42,0.552,0.472,0.904
D15__MF35,il4ra2026,MF,mixed,advanced,0.786,2187,597,0.255,0.693,0.0151,0.0402,0.528,0.652,0.338,0.674,0.326,0.544,0.702,0.874,0.764,0.467,0.309,0.699,0.632,0.509,0.778,0.824,0.133,0.863,0.473,0.277,0.0462,0.00914,0.629,0.144,0.113,0.683,0.827,0.0407,0.833
D3__P196,brunner2024,MF,erythroderma,advanced,0.791,506,134,0.348,0.0771,0.425,0.0455,0.0968,0.182,0.107,0.113,0.0296,0.111,0.0929,0.287,0.425,0.087,0.255,0.217,0.126,0.085,0.403,0.198,0.192,0.375,0.166,0.322,0.0672,0,0.0929,0.237,0.00593,0.291,0.346,0.553,0.605
D3__P192,brunner2024,SS,erythroderma,advanced,0.813,2874,660,0.16,0.278,0.404,0.134,0.157,0.212,0.207,0.185,0.0769,0.247,0.132,0.36,0.493,0.0804,0.185,0.325,0.157,0.122,0.574,0.489,0.158,0.581,0.138,0.394,0.031,0.197,0.0964,0.0299,0.00731,0.42,0.292,0.565,0.66
H__SS4,herrera2021,SS,erythroderma,advanced,0.816,204,46,0.26,0.397,0.172,0.348,0.289,0.373,0.407,0.23,0.319,0.549,0.284,0.49,0.637,0.255,0.353,0.642,0.407,0.216,0.559,0.495,0.284,0.623,0.328,0.422,0.0294,0.588,0.221,0.0343,0.0049,0.657,0.642,0.711,0.309
D1__PGS,chennareddy2025,MF,erythroderma,advanced,0.817,2056,459,0.394,0.53,0.141,0.119,0.19,0.334,0.234,0.18,0.0856,0.325,0.15,0.569,0.582,0.088,0.324,0.437,0.159,0.123,0.541,0.241,0.226,0.644,0.15,0.229,0.017,0.7,0.114,0.0165,0.00632,0.512,0.374,0.662,0.743
D11__P3,brentuximab2026,MF,tumor,advanced,0.827,7596,1589,0.249,0.427,0.221,0.114,0.362,0.344,0.438,0.149,0.145,0.55,0.223,0.207,0.626,0.175,0.264,0.456,0.167,0.156,0.612,0.569,0.207,0.487,0.184,0.431,0.0369,0.692,0.255,0.0113,0.00158,0.0654,0.751,0.744,0.644
D11__P5,brentuximab2026,MF,tumor,advanced,0.827,6550,1370,0.0634,0.111,0.627,0.555,0.371,0.103,0.403,0.2,0.159,0.584,0.105,0.0507,0.562,0.289,0.355,0.499,0.279,0.259,0.522,0.278,0.0238,0.671,0.324,0.723,0.0302,0.711,0.132,0.0159,0.00885,0.553,0.79,0.717,0.765
Li2024_atlas__CTCL3,li2024,MF,mixed,early,0.832,8823,1778,0.0502,0.473,0.462,0,0.395,0.456,0.146,0.23,0.14,0.612,0.435,0.942,0.157,0.203,0.197,0.709,0.39,0.167,0.843,0.0085,0.152,0.914,0.327,0.16,0.00306,0,0.0172,0.0142,0.628,0.0122,0.637,0.00215,0.0034
B5__PT23,buus2025,SS,erythroderma,,0.836,6371,1247,0.28,0.0849,0.225,0.041,0.0377,0.0881,0.178,0.103,0.037,0.199,0.0411,0.341,0.114,0.342,0.137,0.21,0.0672,0.0618,0.211,0.0917,0.0579,0.206,0.103,0.458,0.0228,0.01,0.226,0.618,0.0022,0.225,0.293,0.287,0.309
D15__MF48,il4ra2026,MF,papule,advanced,0.837,3650,709,0.158,0.0532,0.0414,0.0293,0.14,0.302,0.276,0.21,0.0427,0.382,0.136,0.801,0.378,0.232,0.153,0.222,0.148,0.127,0.42,0.498,0.112,0.425,0.303,0.135,0.115,0.000548,0.0299,0.402,0.0723,0.286,0.24,0.0118,0.583
Li2024_atlas__MF28,li2024,MF,tumor,advanced,0.845,10395,1904,0.149,0.142,0.0938,0.107,0.264,0.0955,0.172,0.109,0.163,0.361,0.0721,0.716,0.522,0.0947,0.311,0.318,0.11,0.147,0.557,0.227,0.054,0.588,0.159,0.144,0.0395,0,0.00866,0.0237,0.00202,0.331,0.432,0.675,0.444
Li2024_atlas__PT11,li2024,MF,patch,advanced,0.874,1948,282,0.0272,0.597,0.00257,0.166,0.3,0.304,0.102,0.4,0.248,0.166,0.0857,0.0452,0.161,0.319,0.00462,0.232,0.213,0.13,0.506,0.464,0.0257,0.573,0.141,0.0298,0.00103,0,0.0149,0.967,0.0257,0.516,0.498,0.597,0.0241
Li2024_atlas__CTCL10,li2024,MF,skin_unspecified,early,0.88,1151,157,0.421,0.00348,0.00434,0.884,0.616,0.542,0.566,0.551,0.633,0.395,0.367,0.858,0.235,0.719,0.409,0.401,0.692,0.419,0.503,0.12,0.0156,0.328,0.288,0.00608,0.326,0,0.134,0.00261,0.00608,0.00434,0.00434,0.00608,0.867
D11__P1,brentuximab2026,MF,plaque,advanced,0.883,678,90,0.133,0.193,0.491,0.223,0.198,0.347,0.376,0.156,0.119,0.364,0.0693,0.196,0.504,0.1,0.137,0.305,0.193,0.215,0.431,0.605,0.106,0.363,0.159,0.524,0.0339,0.454,0.114,0.00147,0,0.72,0.643,0.67,0.558
Li2024_atlas__CTCL12,li2024,MF,skin_unspecified,advanced,0.894,2784,331,0.307,0.0305,0.332,0.704,0.0302,0.356,0.813,0.249,0.205,0.564,0.217,0.317,0.443,0.308,0.372,0.191,0.345,0.208,0.335,0.491,0.233,0.298,0.0862,0.916,0.667,0,0.542,0.0154,0.0158,0.456,0.326,0.0154,0.873
D11__P2,brentuximab2026,MF,tumor,advanced,0.899,9473,1063,0.443,0.509,0.176,0.212,0.388,0.193,0.556,0.151,0.17,0.408,0.245,0.694,0.662,0.096,0.265,0.43,0.241,0.127,0.566,0.536,0.341,0.549,0.183,0.246,0.112,0.445,0.113,0.00971,0.00148,0.427,0.554,0.538,0.705
D1__P303,chennareddy2025,MF,tumor,advanced,0.899,1038,116,0.166,0.333,0.121,0.0617,0.573,0.203,0.535,0.28,0.109,0.702,0.328,0.938,0.886,0.0626,0.342,0.676,0.251,0.198,0.722,0.689,0.337,0.674,0.23,0.136,0.053,0.557,0.305,0.00963,0.00482,0.00385,0.00193,0.00289,0.879
Li2024_atlas__PT35,li2024,MF,mixed,advanced,0.919,3502,308,0.189,0.475,0.182,0.0266,0.151,0.347,0.492,0.111,0.0414,0.355,0.0828,0.773,0.448,0.143,0.233,0.353,0.16,0.0557,0.471,0.00857,0.0591,0.411,0.141,0.408,0.0143,0,0.101,0.00286,0,0.647,0.534,0.632,0.842
D15__MF26,il4ra2026,MF,erythroderma,advanced,0.92,3041,265,0.243,0.0214,0.0312,0.195,0.179,0.126,0.436,0.227,0.112,0.28,0.0477,0.0572,0.301,0.355,0.508,0.462,0.231,0.141,0.391,0.271,0.0687,0.473,0.118,0.226,0.0306,0.457,0.105,0.247,0.00362,0.599,0.687,0.598,0.541
D1__P171,chennareddy2025,MF,tumor,advanced,0.932,5216,382,0.457,0.196,0.143,0.143,0.101,0.289,0.222,0.174,0.0633,0.526,0.106,0.338,0.435,0.145,0.473,0.588,0.188,0.112,0.383,0.153,0.294,0.349,0.251,0.373,0.00364,0.126,0.087,0.00652,0.0117,0.741,0.757,0.706,0.357
Li2024_atlas__CTCL9,li2024,MF,skin_unspecified,early,0.954,2216,106,0.136,0.0217,0.0528,0.568,0.112,0.33,0.852,0.171,0.275,0.463,0.345,0.653,0.527,0.19,0.178,0.12,0.294,0.125,0.16,0.302,0.179,0.187,0.0551,0.764,0.689,0,0.29,0.0546,0.00271,0.0135,0.211,0.0289,0.651
Li2024_atlas__CTCL1,li2024,MF,mixed,advanced,0.956,7676,352,0.143,0.5,0.183,0.0106,0.18,0.246,0.121,0.175,0.146,0.479,0.028,0.056,0.384,0.16,0.159,0.561,0.216,0.116,0.493,0.578,0.24,0.608,0.246,0.315,0.000782,0,0.118,0.00534,0.00013,0.414,0.395,0.481,0.488
Li2024_atlas__PT55,li2024,MF,mixed,advanced,0.968,2177,71,0.131,0.77,0,0.11,0.624,0.18,0.112,0.175,0.237,0.00184,0.109,0.763,0.243,0.0684,0.0441,0.44,0.166,0.229,0.536,0.199,0.187,0.559,0.191,0.000919,0.0326,0,0.0873,0.612,0.0799,0.224,0.00643,0.00919,0.00367
Li2024_atlas__CTCL18,li2024,MF,skin_unspecified,advanced,0.982,7004,127,0.204,0.335,0.00286,0.0969,0.512,0.338,0.569,0.3,0.418,0.293,0.365,0.0614,0.34,0.451,0.167,0.262,0.526,0.281,0.194,0.00257,0.116,0.279,0.193,0.271,0.00742,0,0.0281,0.583,0.543,0.00214,0.215,0.00457,0.815
D13__MF17,gaydosik2022,MF,tumor,advanced,0.985,3696,56,0.327,0.0909,0.0179,0.0893,0.249,0.124,0.252,0.172,0.104,0.351,0.0346,0.0468,0.191,0.281,0.259,0.299,0.182,0.0844,0.565,0.148,0.273,0.501,0.272,0.141,0.00189,0.0146,0.0135,0.00406,0.00649,0.573,0.599,0.697,0.718
```

## §6c Per-donor detection, reactive CD4

**What:** share of each patient's reactive (non-malignant) CD4 cells with >= 1 UMI of the gene.

**How:** pct_expr of the CD4_reactive arm; same donors and order as §6b.

**Full table on disk:** `tables/tgt49_donor_pct_reactive.csv`

```csv
donor,study,dx2,lesion,clin_stage_class,burden,n_malignant,n_reactive,CD70,CD9,CPM,COL6A3,CD151,GRN,TFRC,SLC39A6,SLC4A2,TNFRSF25,SLC1A4,LGALS3,CD82,SLC5A3,IL6R,SYPL1,CALU,QSOX1,VDAC1,TNFRSF4,TSPAN2,GPI,TMEM30A,CCR4,TNFRSF8,KIR3DL2,IL2RA,CD7,DPP4,CD5,CD6,CD27,CD2
Li2024_atlas__MF17,li2024,MF,tumor,advanced,0.00837,37,4385,0.0226,0,0.00547,0,0.157,0.138,0.171,0.111,0.0643,0.166,0.0228,0.525,0.573,0.0764,0.432,0.316,0.11,0.155,0.245,0.106,0.0987,0.469,0.243,0.0463,0.000456,0,0.0187,0.081,0.442,0.553,0.706,0.0627,0.763
Li2024_atlas__CTCL7,li2024,MF,mixed,early,0.0869,3658,38432,0.0348,0.00533,0.00765,0.00138,0.0387,0.0571,0.0542,0.0625,0.0327,0.124,0.127,0.265,0.168,0.0766,0.0594,0.204,0.124,0.0292,0.292,0.33,0.0456,0.517,0.104,0.242,0.00172,0,0.0739,0.545,0.0456,0.448,0.347,0.189,0.443
Li2024_atlas__MF27,li2024,MF,tumor,advanced,0.129,107,725,0.0745,0.0069,0.0648,0.00414,0.119,0.102,0.128,0.0966,0.0386,0.259,0.0662,0.286,0.412,0.153,0.143,0.241,0.0952,0.142,0.335,0.338,0.0648,0.377,0.189,0.177,0.0152,0,0.167,0.681,0.0469,0.458,0.47,0.407,0.735
Li2024_atlas__CTCL4,li2024,MF,mixed,early,0.136,1241,7906,0.0323,0.0271,0.00987,0.000632,0.102,0.0765,0.0517,0.0665,0.0302,0.101,0.0493,0.231,0.177,0.0645,0.0615,0.226,0.101,0.0376,0.259,0.27,0.0245,0.47,0.118,0.189,0.00847,0,0.0933,0.706,0.034,0.22,0.197,0.213,0.417
D15__MF28,il4ra2026,MF,tumor,advanced,0.151,123,691,0.0478,0.0289,0.126,0.0232,0.0564,0.137,0.255,0.194,0.0564,0.168,0.143,0.172,0.23,0.42,0.113,0.259,0.247,0.0897,0.331,0.346,0.0304,0.443,0.279,0.314,0.0101,0.00724,0.398,0.698,0.0912,0.509,0.572,0.113,0.654
D10__JONAK1,jonak2021,MF,plaque,early,0.153,348,1920,0.00573,0.0302,0.00937,0.0281,0.0667,0.0922,0.0568,0.037,0.00833,0.0797,0.00937,0.136,0.283,0.0208,0.138,0.238,0.0531,0.0333,0.213,0.143,0.0156,0.226,0.0776,0.16,0.00313,0.00313,0.0839,0.424,0.0323,0.176,0.144,0.314,0.584
D13__MF21,gaydosik2022,MF,tumor,advanced,0.177,118,547,0.106,0.124,0.0146,0.0402,0.0914,0.0804,0.053,0.0567,0.0293,0.0768,0.0311,0.26,0.278,0.0548,0.0439,0.0457,0.0238,0.0366,0.146,0.179,0.0402,0.117,0.0603,0.0183,0.00731,0.00183,0.11,0.263,0.0293,0.108,0.141,0.181,0.366
D3__P105,brunner2024,MF,erythroderma,advanced,0.185,117,516,0.0581,0.095,0.062,0.0581,0.0329,0.0504,0.143,0.0872,0.0213,0.114,0.0484,0.246,0.246,0.0698,0.196,0.163,0.0504,0.031,0.171,0.163,0.0233,0.149,0.161,0.211,0.0329,0.0194,0.0678,0.153,0.0155,0.101,0.178,0.165,0.388
D15__MF36,il4ra2026,MF,mixed,advanced,0.221,620,2182,0.0706,0.0348,0.0779,0.0247,0.0454,0.0825,0.387,0.127,0.0385,0.18,0.19,0.396,0.238,0.322,0.128,0.194,0.125,0.0802,0.293,0.394,0.0151,0.319,0.11,0.231,0.0541,0.0124,0.389,0.535,0.0403,0.419,0.44,0.24,0.531
D8__P90,alkon2024,MF,patch,early,0.227,68,232,0.0172,0.0776,0.00431,0.0388,0.121,0.134,0.155,0.069,0.00862,0.0259,0.0345,0.284,0.254,0.0474,0.056,0.172,0.0948,0.0129,0.241,0.207,0.0474,0.306,0.0302,0.116,0.00431,0.00431,0.164,0.599,0.0216,0.203,0.151,0.241,0.569
D15__MF24,il4ra2026,MF,tumor,advanced,0.247,643,1958,0.0332,0.00766,0.0541,0.0511,0.0659,0.07,0.225,0.0802,0.0322,0.176,0.0541,0.106,0.215,0.172,0.0899,0.124,0.0689,0.0664,0.183,0.258,0.025,0.24,0.047,0.0572,0.0184,0.0112,0.316,0.616,0.0281,0.198,0.2,0.279,0.455
D11__P4,brentuximab2026,MF,plaque,advanced,0.254,174,512,0.102,0.176,0.0996,0.0605,0.115,0.133,0.176,0.0996,0.0547,0.205,0.145,0.568,0.402,0.102,0.0918,0.205,0.0996,0.0723,0.299,0.295,0.0723,0.303,0.0723,0.143,0.0898,0.0195,0.174,0.268,0.0703,0.309,0.4,0.223,0.611
D15__MF34,il4ra2026,MF,tumor,advanced,0.264,248,692,0.0419,0.0145,0.0145,0.00723,0.0737,0.0737,0.155,0.12,0.026,0.185,0.147,0.165,0.129,0.253,0.134,0.233,0.0939,0.0997,0.253,0.182,0.0202,0.316,0.149,0.136,0.00145,0.013,0.263,0.6,0.184,0.228,0.392,0.0997,0.425
Li2024_atlas__CTCL6,li2024,MF,mixed,early,0.277,3937,10298,0.0205,0.00738,0.028,0.00398,0.0654,0.0799,0.077,0.0971,0.0393,0.19,0.095,0.229,0.21,0.114,0.11,0.245,0.15,0.0499,0.33,0.339,0.0384,0.56,0.164,0.43,0.00398,0,0.0852,0.569,0.0763,0.478,0.423,0.243,0.433
Li2024_atlas__CTCL14,li2024,MF,skin_unspecified,early,0.288,63,156,0.0321,0.192,0.0385,0.0449,0.16,0.147,0.25,0.128,0.135,0.179,0.0192,0.218,0.295,0.276,0.135,0.00641,0.0833,0.0577,0.115,0.0897,0.385,0.0321,0.0192,0.167,0,0,0.167,0.0641,0.0256,0.199,0.16,0.109,0.288
D7__P73,rindler2021_mc,MF,plaque,advanced,0.327,189,389,0.0257,0.0283,0.00514,0.0103,0.0566,0.054,0.036,0.0411,0.00514,0.0694,0.00514,0.249,0.141,0.0257,0.036,0.18,0.0488,0,0.175,0.0925,0.0386,0.224,0.0771,0.09,0.00257,0.0103,0.0771,0.517,0.0463,0.162,0.131,0.314,0.447
D7__P84,rindler2021_mc,MF,plaque,advanced,0.327,1892,3889,0.0507,0.0548,0.0198,0.0247,0.0481,0.0645,0.0609,0.0627,0.0149,0.0388,0.0157,0.248,0.277,0.0368,0.0823,0.188,0.054,0.0144,0.183,0.118,0.0293,0.217,0.0586,0.107,0.00309,0.00874,0.0869,0.338,0.0278,0.237,0.21,0.249,0.492
H__SS1,herrera2021,SS,patch,advanced,0.33,36,73,0.164,0.233,0.0274,0.219,0.11,0.178,0.219,0.137,0.0685,0.288,0.137,0.616,0.685,0.247,0.247,0.397,0.137,0.0822,0.425,0.507,0.26,0.397,0.205,0.26,0.0411,0.0137,0.11,0.452,0.0548,0.37,0.384,0.466,0.603
Li2024_atlas__PT56,li2024,MF,mixed,advanced,0.343,1292,2478,0.0533,0.0513,0.0137,0.0262,0.0924,0.171,0.105,0.0456,0.0335,0.161,0.0464,0.295,0.456,0.0492,0.131,0.188,0.0783,0.0642,0.312,0.285,0.0839,0.341,0.1,0.111,0.0569,0,0.172,0.441,0.0517,0.319,0.356,0.325,0.659
Li2024_atlas__MF22,li2024,MF,plaque,early,0.343,300,575,0.0365,0.00348,0.0104,0.0104,0.0957,0.0696,0.127,0.111,0.0922,0.318,0.0713,0.348,0.369,0.0991,0.217,0.341,0.073,0.134,0.381,0.363,0.0991,0.449,0.209,0.163,0.0296,0,0.203,0.715,0.123,0.423,0.468,0.313,0.713
D13__MF25,gaydosik2022,unresolved,erythroderma,advanced,0.354,99,181,0.0331,0.0939,0.0331,0.0718,0.0884,0.0608,0.199,0.0442,0.0552,0.149,0.0663,0.486,0.326,0.116,0.0884,0.0939,0.105,0.0442,0.232,0.11,0.0663,0.293,0.0442,0.0442,0.00552,0.0221,0.0331,0.304,0.00552,0.199,0.315,0.171,0.514
D1__P204,chennareddy2025,MF,tumor,advanced,0.375,3444,5741,0.0822,0.0735,0.0202,0.0366,0.0817,0.0721,0.0671,0.0627,0.0333,0.12,0.0315,0.206,0.271,0.0571,0.128,0.168,0.0629,0.0582,0.211,0.118,0.0368,0.231,0.146,0.133,0.00453,0.0462,0.124,0.479,0.0383,0.177,0.242,0.307,0.52
Li2024_atlas__CTCL2,li2024,MF,mixed,early,0.382,1357,2200,0.00727,0.0173,0.0136,0,0.0759,0.108,0.0368,0.0895,0.0309,0.17,0.0373,0.238,0.132,0.0464,0.105,0.276,0.0955,0.0523,0.343,0.192,0.0409,0.544,0.0836,0.335,0,0,0.0295,0.8,0.0691,0.327,0.337,0.311,0.578
Li2024_atlas__PT50,li2024,MF,mixed,early,0.441,1464,1858,0.0409,0.0296,0.0452,0.0339,0.0581,0.085,0.21,0.0888,0.0457,0.284,0.0581,0.174,0.328,0.0969,0.167,0.198,0.0904,0.0517,0.333,0.358,0.0802,0.308,0.175,0.214,0.0172,0,0.235,0.596,0.0404,0.292,0.36,0.299,0.695
D13__MF18,gaydosik2022,MF,tumor,advanced,0.469,826,935,0.122,0.0171,0.121,0.0706,0.101,0.119,0.247,0.0856,0.077,0.283,0.0492,0.219,0.368,0.175,0.121,0.229,0.105,0.0749,0.36,0.367,0.0781,0.396,0.145,0.061,0.0267,0.0246,0.113,0.484,0.0353,0.369,0.446,0.348,0.694
D6__MF2,gaydosik2019,SS,erythroderma,advanced,0.489,677,708,0.0466,0.398,0.0254,0.00989,0.0452,0.0466,0.192,0.0833,0.0339,0.0777,0.134,0.631,0.205,0.154,0.0621,0.244,0.0692,0.0311,0.298,0.181,0.0381,0.0989,0.0636,0.00847,0.0113,0.048,0.153,0.355,0.00565,0.0989,0.212,0.202,0.53
D8__P65,alkon2024,MF,patch,early,0.496,615,626,0.0319,0.166,0.00799,0.0272,0.0511,0.0479,0.0559,0.0383,0.00479,0.0623,0.0224,0.308,0.185,0.0415,0.0671,0.144,0.0288,0.0256,0.176,0.176,0.024,0.139,0.0799,0.0927,0.0016,0.00479,0.0831,0.214,0.0224,0.129,0.0927,0.222,0.363
D8__P138,alkon2024,MF,patch,early,0.505,627,614,0.0489,0.109,0.0212,0.0212,0.0635,0.0586,0.121,0.0456,0.0228,0.181,0.0489,0.342,0.251,0.0668,0.0684,0.166,0.0668,0.0358,0.233,0.204,0.0489,0.181,0.0863,0.0945,0.0179,0.00977,0.119,0.171,0.0212,0.119,0.147,0.197,0.432
D13__MF19,gaydosik2022,unresolved,tumor,advanced,0.518,253,235,0.123,0.328,0.0553,0.0936,0.0894,0.162,0.217,0.0681,0.106,0.149,0.102,0.583,0.43,0.209,0.17,0.251,0.0894,0.132,0.383,0.353,0.0638,0.366,0.123,0.145,0.0426,0.017,0.221,0.587,0.034,0.34,0.387,0.404,0.511
Li2024_atlas__MF30,li2024,MF,tumor,advanced,0.549,2342,1923,0.0317,0.0208,0.0166,0.00676,0.0634,0.103,0.0598,0.0863,0.0354,0.244,0.0187,0.134,0.29,0.0593,0.184,0.388,0.0842,0.0775,0.296,0.144,0.04,0.411,0.189,0.0666,0.00416,0,0.0608,0.732,0.122,0.498,0.498,0.525,0.711
D1__P312,chennareddy2025,MF,plaque,advanced,0.566,717,550,0.04,0.0273,0.0164,0.0327,0.0545,0.0745,0.105,0.04,0.02,0.0982,0.0545,0.285,0.247,0.0309,0.0727,0.169,0.0527,0.0345,0.253,0.251,0.06,0.191,0.0636,0.109,0.0273,0.0327,0.185,0.449,0.0364,0.233,0.187,0.32,0.498
D6__SC157,gaydosik2019,MF,tumor,advanced,0.572,791,593,0.118,0.0759,0.0287,0.0202,0.116,0.135,0.0911,0.0877,0.0489,0.159,0.0624,0.211,0.182,0.25,0.0337,0.233,0.118,0.0573,0.396,0.371,0.0506,0.18,0.0691,0.0523,0.0118,0.027,0.142,0.723,0.032,0.255,0.268,0.251,0.632
D15__MF30,il4ra2026,MF,erythroderma,advanced,0.584,497,354,0.311,0.203,0.189,0.105,0.373,0.54,0.523,0.534,0.333,0.588,0.655,0.619,0.782,0.531,0.421,0.653,0.429,0.367,0.757,0.672,0.141,0.856,0.585,0.562,0.161,0.316,0.655,0.757,0.322,0.749,0.709,0.398,0.836
H__MFIV1,herrera2021,MF,plaque,advanced,0.62,44,27,0.0741,0.111,0.037,0.778,0.259,0.111,0.148,0.0741,0.0741,0.148,0.111,0.519,0.481,0.148,0.148,0.296,0.148,0.222,0.481,0.407,0.296,0.37,0.111,0.222,0.0741,0.037,0.185,0.667,0.0741,0.296,0.519,0.333,0.63
Li2024_atlas__PT52,li2024,MF,mixed,advanced,0.643,160,89,0.0787,0.135,0.0225,0.0337,0.112,0.124,0.157,0.0225,0.101,0.337,0.0337,0.382,0.494,0.0899,0.236,0.135,0.101,0.0562,0.326,0.213,0.124,0.292,0.0899,0.0562,0.0225,0,0.0787,0.202,0.0562,0.191,0.371,0.135,0.708
Li2024_atlas__CTCL5,li2024,MF,mixed,advanced,0.649,7532,4073,0.0412,0.0162,0.0435,0.0027,0.0592,0.101,0.0808,0.0722,0.0184,0.296,0.0977,0.308,0.29,0.122,0.105,0.226,0.131,0.0324,0.438,0.456,0.0469,0.61,0.0854,0.246,0.00589,0,0.132,0.813,0.0459,0.399,0.316,0.296,0.654
Li2024_atlas__PT47,li2024,MF,mixed,advanced,0.65,307,165,0.0606,0.0364,0.0242,0.0667,0.097,0.152,0.182,0.097,0.0667,0.279,0.0242,0.279,0.364,0.0848,0.194,0.218,0.0788,0.0727,0.321,0.218,0.0727,0.285,0.17,0.139,0.0545,0,0.152,0.327,0.0424,0.297,0.376,0.382,0.582
Li2024_atlas__PT53,li2024,MF,mixed,advanced,0.658,1417,735,0.0109,0.00272,0.0122,0.0218,0.0313,0.0735,0.165,0.0626,0.0408,0.244,0.0381,0.182,0.331,0.083,0.18,0.215,0.0667,0.0599,0.261,0.261,0.0449,0.301,0.186,0.124,0.0177,0,0.107,0.57,0.0544,0.241,0.309,0.278,0.578
B5__PT09,buus2025,SS,erythroderma,,0.664,10427,5273,0.0432,0.109,0.127,0.00607,0.0586,0.0554,0.265,0.0872,0.0745,0.14,0.178,0.54,0.367,0.141,0.235,0.242,0.0804,0.137,0.305,0.271,0.0533,0.417,0.13,0.232,0.0339,0.0252,0.25,0.566,0.0298,0.356,0.441,0.412,0.658
Li2024_atlas__CTCL13,li2024,MF,skin_unspecified,advanced,0.674,5962,2884,0.0877,0.0648,0.0409,0.212,0.061,0.181,0.45,0.0801,0.116,0.443,0.0919,0.183,0.341,0.224,0.162,0.0593,0.152,0.133,0.0468,0.188,0.0586,0.105,0.0548,0.369,0.0489,0,0.295,0.469,0.0808,0.358,0.522,0.418,0.53
D8__P163,alkon2024,MF,patch,early,0.679,881,417,0.048,0.0624,0.0168,0.00959,0.036,0.0504,0.048,0.0432,0.0048,0.0959,0.012,0.314,0.132,0.0264,0.048,0.108,0.0216,0.0216,0.192,0.103,0.048,0.146,0.0839,0.0576,0.00719,0.0024,0.0456,0.297,0.0767,0.115,0.151,0.0839,0.393
Li2024_atlas__CTCL17,li2024,MF,skin_unspecified,advanced,0.712,203,82,0.39,0.415,0.0488,0.0732,0.28,0.329,0.354,0.159,0.134,0.39,0.207,0.451,0.256,0.415,0.0122,0.256,0.305,0.11,0.232,0.0488,0.232,0.207,0.0976,0.0854,0,0,0.061,0.256,0.695,0.061,0.402,0.0488,0.854
Li2024_atlas__MF21,li2024,MF,tumor,advanced,0.713,1761,708,0.0862,0.0339,0.0254,0.00565,0.192,0.13,0.205,0.138,0.089,0.278,0.219,0.318,0.411,0.256,0.153,0.349,0.147,0.164,0.523,0.355,0.116,0.516,0.267,0.251,0.0212,0,0.282,0.816,0.127,0.422,0.477,0.364,0.737
D8__P150,alkon2024,MF,patch,early,0.724,2393,912,0.00987,0.0143,0.0121,0.00877,0.0285,0.0428,0.0482,0.0307,0.00877,0.103,0.00987,0.161,0.226,0.0285,0.138,0.113,0.0406,0.0143,0.0943,0.106,0.0175,0.147,0.0779,0.122,0.00219,0,0.0625,0.248,0.0439,0.211,0.252,0.227,0.453
D7__MF311,rindler2021_mc,MF,patch,advanced,0.748,5693,1919,0.0292,0.0234,0.0151,0.0135,0.0438,0.036,0.061,0.0313,0.00834,0.0532,0.0182,0.197,0.164,0.0302,0.0573,0.0917,0.0406,0.0219,0.167,0.0834,0.0318,0.172,0.0537,0.0375,0.00208,0.0229,0.0542,0.324,0.0318,0.108,0.135,0.184,0.401
Li2024_atlas__CTCL8,li2024,MF,mixed,early,0.751,7379,2447,0.04,0.0217,0.0045,0.000817,0.036,0.00572,0.0286,0.0327,0.0131,0.0429,0.027,0.245,0.0944,0.0478,0.0131,0.153,0.0548,0.00286,0.131,0.137,0.0217,0.112,0.0499,0.145,0,0,0.067,0.612,0.0262,0.0609,0.0568,0.131,0.248
D7__MF309,rindler2021_mc,MF,patch,advanced,0.755,5395,1748,0.0383,0.0286,0.0921,0.0355,0.0532,0.0584,0.109,0.0492,0.0172,0.0549,0.0132,0.161,0.299,0.0372,0.0721,0.146,0.0641,0.0189,0.275,0.212,0.0286,0.281,0.0761,0.0932,0.0137,0.0109,0.102,0.332,0.0137,0.199,0.087,0.279,0.506
D6__SC158,gaydosik2019,MF,plaque,advanced,0.756,306,99,0.131,0.354,0.0202,0,0.0909,0.0707,0.0707,0.0303,0.0202,0.121,0.0707,0.475,0.293,0.101,0.0505,0.202,0.0505,0.0707,0.343,0.263,0.0202,0.152,0.0808,0.0101,0.0101,0,0.162,0.707,0,0.121,0.192,0.404,0.717
D11__P6,brentuximab2026,MF,plaque,advanced,0.769,4575,1376,0.0807,0.109,0.0218,0.0618,0.107,0.112,0.145,0.069,0.0371,0.128,0.0734,0.291,0.219,0.0603,0.0916,0.153,0.1,0.069,0.273,0.126,0.0538,0.302,0.118,0.155,0.00363,0.0305,0.0843,0.543,0.056,0.263,0.341,0.265,0.522
Li2024_atlas__CTCL16,li2024,MF,skin_unspecified,early,0.779,1037,294,0.0408,0.085,0.167,0.156,0.051,0.133,0.374,0.0918,0.0782,0.422,0.0578,0.15,0.276,0.241,0.173,0.0408,0.126,0.109,0.133,0.218,0.0476,0.0884,0.0102,0.514,0.0476,0,0.259,0.289,0.0204,0.32,0.435,0.364,0.765
D15__MF35,il4ra2026,MF,mixed,advanced,0.786,2187,597,0.0637,0.111,0.0285,0.0101,0.209,0.208,0.176,0.231,0.072,0.286,0.394,0.559,0.332,0.251,0.0871,0.295,0.208,0.147,0.441,0.315,0.0804,0.663,0.345,0.0804,0.00168,0.0201,0.251,0.821,0.322,0.595,0.745,0.288,0.697
D3__P196,brunner2024,MF,erythroderma,advanced,0.791,506,134,0.0373,0.0597,0.0149,0.0149,0.0821,0.119,0.0522,0.0821,0.0224,0.0522,0.0224,0.291,0.276,0.0597,0.0672,0.157,0.0597,0.0448,0.299,0.119,0.0896,0.269,0.0896,0.0373,0,0,0.0522,0.463,0.112,0.209,0.261,0.142,0.53
D3__P192,brunner2024,SS,erythroderma,advanced,0.813,2874,660,0.1,0.153,0.0348,0.102,0.145,0.0955,0.0758,0.0788,0.05,0.0727,0.0606,0.403,0.291,0.0712,0.0379,0.111,0.0955,0.0803,0.352,0.145,0.102,0.306,0.141,0.0485,0.0121,0.0288,0.0561,0.524,0.0394,0.144,0.206,0.238,0.582
H__SS4,herrera2021,SS,erythroderma,advanced,0.816,204,46,0.087,0.174,0.0435,0.196,0.239,0.196,0.217,0.0652,0.087,0.239,0.174,0.435,0.283,0.087,0.152,0.326,0.087,0.13,0.261,0.217,0.152,0.457,0.174,0.0652,0,0.087,0.217,0.609,0.217,0.261,0.37,0.413,0.783
D1__PGS,chennareddy2025,MF,erythroderma,advanced,0.817,2056,459,0.142,0.161,0.0153,0.0414,0.174,0.126,0.0545,0.0566,0.0414,0.0479,0.0479,0.438,0.351,0.0305,0.0871,0.172,0.0697,0.0479,0.327,0.161,0.0784,0.388,0.113,0.0959,0.0109,0.118,0.0893,0.471,0.0305,0.187,0.207,0.427,0.728
D11__P3,brentuximab2026,MF,tumor,advanced,0.827,7596,1589,0.0459,0.0837,0.0283,0.0151,0.0957,0.104,0.137,0.0497,0.0327,0.171,0.073,0.169,0.231,0.0636,0.0636,0.193,0.0717,0.0585,0.296,0.228,0.0352,0.255,0.0806,0.134,0.00755,0.0566,0.128,0.62,0.0699,0.243,0.345,0.244,0.445
D11__P5,brentuximab2026,MF,tumor,advanced,0.827,6550,1370,0.0204,0.0241,0.0869,0.0672,0.0686,0.113,0.229,0.0766,0.0489,0.209,0.0423,0.0927,0.212,0.0876,0.174,0.26,0.119,0.0847,0.357,0.177,0.0409,0.357,0.168,0.208,0.00876,0.112,0.122,0.715,0.163,0.407,0.56,0.376,0.634
Li2024_atlas__CTCL3,li2024,MF,mixed,early,0.832,8823,1778,0.0219,0.045,0.0231,0.00112,0.11,0.114,0.0641,0.0793,0.0315,0.165,0.0467,0.234,0.162,0.0624,0.0619,0.358,0.132,0.0596,0.438,0.314,0.054,0.733,0.12,0.223,0.000562,0,0.036,0.808,0.0506,0.426,0.403,0.298,0.537
B5__PT23,buus2025,SS,erythroderma,,0.836,6371,1247,0.0529,0.123,0.0168,0.00321,0.0377,0.0273,0.123,0.0794,0.0337,0.0377,0.069,0.348,0.171,0.134,0.0634,0.172,0.0561,0.069,0.174,0.176,0.0401,0.224,0.1,0.168,0.00401,0.107,0.107,0.365,0.0353,0.218,0.274,0.0882,0.417
D15__MF48,il4ra2026,MF,papule,advanced,0.837,3650,709,0.0367,0.00846,0.0183,0.00705,0.0381,0.055,0.162,0.0776,0.0282,0.203,0.0508,0.159,0.164,0.213,0.0719,0.189,0.106,0.0578,0.258,0.327,0.0268,0.254,0.079,0.205,0.00987,0.00141,0.145,0.584,0.0437,0.396,0.417,0.22,0.511
Li2024_atlas__MF28,li2024,MF,tumor,advanced,0.845,10395,1904,0.0683,0.0446,0.0378,0.0184,0.106,0.0767,0.0704,0.0657,0.0452,0.207,0.0473,0.386,0.35,0.0924,0.152,0.298,0.0704,0.0956,0.393,0.192,0.0509,0.451,0.141,0.0961,0.0137,0,0.0672,0.631,0.0867,0.367,0.386,0.571,0.706
Li2024_atlas__PT11,li2024,MF,patch,advanced,0.874,1948,282,0.0142,0.0284,0.00355,0.00709,0.0603,0.078,0.0248,0.039,0.0248,0.202,0.0603,0.0922,0.149,0.0603,0.0603,0.0745,0.0284,0.0248,0.216,0.266,0.039,0.191,0.0532,0.0674,0,0,0.0355,0.723,0.0177,0.152,0.181,0.27,0.397
Li2024_atlas__CTCL10,li2024,MF,skin_unspecified,early,0.88,1151,157,0.146,0.0446,0.0127,0.376,0.217,0.217,0.465,0.236,0.236,0.389,0.166,0.389,0.28,0.338,0.178,0.134,0.299,0.21,0.21,0.051,0.0764,0.178,0.127,0.229,0.134,0,0.178,0.35,0.0573,0.248,0.35,0.376,0.783
D11__P1,brentuximab2026,MF,plaque,advanced,0.883,678,90,0.0778,0.156,0.0889,0,0.0556,0.133,0.156,0.1,0.0778,0.133,0.244,0.433,0.289,0.0556,0.0333,0.244,0.0778,0.1,0.289,0.489,0.0444,0.4,0.1,0.1,0.0333,0.0111,0.533,0.433,0.0444,0.222,0.2,0.189,0.467
Li2024_atlas__CTCL12,li2024,MF,skin_unspecified,advanced,0.894,2784,331,0.0937,0.0363,0.0363,0.218,0.0211,0.163,0.411,0.121,0.133,0.278,0.142,0.205,0.263,0.236,0.0937,0.0483,0.163,0.0997,0.148,0.13,0.172,0.0785,0.0544,0.281,0.115,0,0.16,0.317,0.0514,0.233,0.35,0.32,0.761
D11__P2,brentuximab2026,MF,tumor,advanced,0.899,9473,1063,0.0536,0.0743,0.0301,0.0376,0.104,0.105,0.183,0.079,0.0405,0.198,0.0781,0.211,0.242,0.064,0.123,0.244,0.0988,0.063,0.297,0.231,0.0564,0.304,0.0903,0.16,0.0151,0.0357,0.116,0.622,0.0837,0.316,0.449,0.373,0.616
D1__P303,chennareddy2025,MF,tumor,advanced,0.899,1038,116,0.069,0.0517,0.0172,0.0172,0.147,0.069,0.069,0.069,0.00862,0.112,0.0345,0.353,0.224,0.0259,0.0431,0.147,0.069,0.0345,0.31,0.259,0.121,0.293,0.103,0.0345,0,0.00862,0.172,0.784,0.0517,0.155,0.216,0.216,0.595
Li2024_atlas__PT35,li2024,MF,mixed,advanced,0.919,3502,308,0.00974,0.0487,0.0162,0.00974,0.12,0.0909,0.0487,0.0747,0.0162,0.0487,0.039,0.338,0.127,0.0552,0.0162,0.0584,0.0617,0.0162,0.292,0.0974,0.0357,0.214,0.0877,0.0325,0,0,0.00649,0.883,0,0.0682,0.0779,0.175,0.231
D15__MF26,il4ra2026,MF,erythroderma,advanced,0.92,3041,265,0.0943,0.0755,0.0566,0.0868,0.125,0.0868,0.238,0.143,0.083,0.109,0.14,0.423,0.506,0.343,0.121,0.196,0.177,0.109,0.419,0.245,0.0642,0.517,0.0981,0.136,0.0453,0.0151,0.158,0.415,0.0792,0.374,0.543,0.309,0.653
D1__P171,chennareddy2025,MF,tumor,advanced,0.932,5216,382,0.113,0.12,0.0105,0.0288,0.0864,0.0942,0.0916,0.0759,0.034,0.113,0.0471,0.262,0.243,0.0524,0.0995,0.207,0.102,0.0602,0.343,0.0942,0.0995,0.249,0.141,0.0759,0,0.0262,0.0419,0.492,0.0419,0.217,0.293,0.351,0.662
Li2024_atlas__CTCL9,li2024,MF,skin_unspecified,early,0.954,2216,106,0.00943,0.0189,0.0283,0.255,0.066,0.123,0.443,0.066,0.0849,0.406,0.0755,0.292,0.198,0.132,0.142,0.0943,0.104,0.151,0.0849,0.0943,0.0755,0.066,0.0283,0.491,0.16,0,0.104,0.283,0.00943,0.264,0.377,0.142,0.642
Li2024_atlas__CTCL1,li2024,MF,mixed,advanced,0.956,7676,352,0.102,0.0568,0.0852,0.00852,0.224,0.159,0.105,0.153,0.0682,0.276,0.125,0.293,0.29,0.0966,0.0824,0.287,0.148,0.0938,0.48,0.247,0.0369,0.582,0.216,0.182,0.0227,0,0.134,0.719,0.0426,0.307,0.429,0.344,0.685
Li2024_atlas__PT55,li2024,MF,mixed,advanced,0.968,2177,71,0.0704,0.0563,0.0704,0.0282,0.0845,0.141,0.169,0.0423,0.0563,0.268,0.0563,0.127,0.324,0,0.127,0.211,0.0423,0.113,0.282,0.38,0.0141,0.352,0.141,0.127,0.0563,0,0.169,0.563,0.0141,0.324,0.282,0.352,0.62
Li2024_atlas__CTCL18,li2024,MF,skin_unspecified,advanced,0.982,7004,127,0.0709,0.173,0.0157,0.15,0.283,0.157,0.291,0.189,0.213,0.622,0.165,0.063,0.315,0.394,0.244,0.118,0.276,0.213,0.0709,0.102,0.0866,0.157,0.0945,0.378,0.0157,0,0.0394,0.402,0.268,0.346,0.559,0.102,0.669
D13__MF17,gaydosik2022,MF,tumor,advanced,0.985,3696,56,0.0893,0.0893,0.0536,0.0179,0.143,0.0357,0.125,0.0714,0.143,0.125,0.179,0.625,0.482,0.107,0.0536,0.232,0.179,0.0536,0.429,0.232,0.0536,0.446,0.214,0.0536,0.0357,0.196,0.161,0.839,0.0714,0.286,0.304,0.357,0.821
```

## §6d Per-donor detection, healthy-skin CD4

**What:** share of each healthy donor's CD4 cells with >= 1 UMI of the gene: the T-cell on-target off-tumour cost.

**How:** pct_expr of the CD4_HC arm (clin_dx == HC); n_cells = all T/NK cells of the donor in the object. Different people, studies and chemistries from the patients.

**Full table on disk:** `tables/tgt49_donor_pct_healthy.csv`

```csv
donor,study,dx,lesion,n_cells,CD70,CD9,CPM,COL6A3,CD151,GRN,TFRC,SLC39A6,SLC4A2,TNFRSF25,SLC1A4,LGALS3,CD82,SLC5A3,IL6R,SYPL1,CALU,QSOX1,VDAC1,TNFRSF4,TSPAN2,GPI,TMEM30A,CCR4,TNFRSF8,KIR3DL2,IL2RA,CD7,DPP4,CD5,CD6,CD27,CD2
D1__P121,chennareddy2025,HC,healthy_skin,83,0,0.0938,0,0.0312,0.0625,0.0156,0.172,0.0469,0.0156,0,0,0.5,0.297,0,0.0469,0.0781,0.0156,0.0156,0.109,0.172,0.0625,0.109,0.0625,0.0781,0,0,0.0781,0.172,0.0469,0.0469,0.0469,0.156,0.344
D1__P116,chennareddy2025,HC,healthy_skin,485,0.0248,0.156,0.00677,0.0158,0.129,0.079,0.144,0.0993,0.0316,0.129,0.0158,0.354,0.366,0.0519,0.219,0.293,0.0767,0.0632,0.217,0.126,0.12,0.278,0.185,0.142,0.00451,0.0135,0.111,0.287,0.115,0.266,0.341,0.248,0.585
D1__P115,chennareddy2025,HC,healthy_skin,605,0.024,0.109,0.0174,0.0218,0.0675,0.0893,0.137,0.0588,0.0131,0.0741,0.0392,0.46,0.242,0.0458,0.157,0.227,0.0763,0.0588,0.298,0.115,0.085,0.211,0.133,0.0784,0.00218,0.0109,0.0828,0.211,0.0675,0.168,0.229,0.113,0.418
D1__P112,chennareddy2025,HC,healthy_skin,269,0.0456,0.12,0.00415,0.00415,0.108,0.0913,0.137,0.0664,0.0207,0.0622,0.029,0.51,0.349,0.0166,0.149,0.199,0.104,0.0373,0.212,0.195,0.141,0.245,0.116,0.0954,0,0.0083,0.129,0.303,0.12,0.187,0.216,0.187,0.494
D6__NOR4,gaydosik2019,HC,healthy_skin,73,0.0492,0.18,0.0328,0,0.0492,0.082,0.082,0,0.0164,0.0984,0.0492,0.361,0.0656,0.0984,0.0984,0.23,0.0328,0.0328,0.148,0.164,0.082,0.0656,0.0328,0.0656,0,0.0164,0.082,0.18,0.0492,0.0492,0.246,0.148,0.443
D6__NOR2,gaydosik2019,HC,healthy_skin,66,0,0.224,0.0204,0.0204,0.0612,0.0204,0.122,0.0408,0,0.102,0.0204,0.327,0.0816,0.0408,0.102,0.245,0.0204,0.0408,0.347,0.245,0.0204,0,0.0612,0.0204,0,0,0.163,0.347,0.0408,0.102,0.184,0.0816,0.408
D6__NOR1,gaydosik2019,HC,healthy_skin,62,0.0784,0.098,0.0196,0,0,0.0196,0.118,0.0196,0,0.098,0.0392,0.196,0.176,0.0588,0.0196,0.0784,0.0392,0.0196,0.176,0.137,0.0196,0.0784,0.0392,0,0,0,0.0784,0.157,0.0196,0.0784,0.137,0.176,0.392
D6__NOR3,gaydosik2019,HC,healthy_skin,97,0,0.138,0.025,0.0125,0.075,0.0625,0.0125,0.025,0.0125,0.0875,0.025,0.312,0.138,0.075,0.075,0.188,0.025,0,0.275,0.212,0.0125,0.0375,0.025,0.0625,0,0,0.15,0.275,0.0625,0.0375,0.0875,0.0375,0.463
D13__SC327_HTO1,gaydosik2022,HC,healthy_skin,28,0.0357,0.107,0,0.0357,0.143,0.0357,0,0.0714,0,0.214,0,0.393,0.357,0,0.0714,0.179,0.107,0,0.25,0.0714,0,0.286,0.0357,0,0,0,0,0.179,0,0.107,0.107,0,0.464
D13__SC374_HTO9,gaydosik2022,HC,healthy_skin,230,0.0236,0.137,0.00472,0.0377,0.0802,0.0613,0.113,0.0849,0.0377,0.0943,0.033,0.382,0.316,0.0566,0.104,0.142,0.0425,0.0755,0.175,0.137,0.0377,0.255,0.066,0.0708,0,0.0283,0.104,0.217,0.0566,0.104,0.17,0.0896,0.344
D13__SC374_HTO10,gaydosik2022,HC,healthy_skin,935,0.0371,0.119,0.0217,0.0332,0.0537,0.0435,0.16,0.0512,0.023,0.132,0.0205,0.495,0.318,0.0499,0.109,0.211,0.0524,0.0486,0.189,0.116,0.0396,0.311,0.0742,0.0627,0.00767,0.00256,0.0754,0.247,0.0537,0.16,0.21,0.134,0.467
D13__SC327_HTO2,gaydosik2022,HC,healthy_skin,225,0.0168,0.0559,0.0112,0.0168,0.0894,0.106,0.0894,0.0447,0.0168,0.134,0.0279,0.475,0.335,0.067,0.0726,0.251,0.0615,0.0335,0.291,0.14,0.0335,0.251,0.0335,0.0726,0,0.0223,0.0838,0.324,0.0894,0.168,0.285,0.173,0.48
D14__SC354,gaydosik2023,HC,healthy_skin,111,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
D14__SC353,gaydosik2023,HC,healthy_skin,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
D14__SC348,gaydosik2023,HC,healthy_skin,494,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
D14__SC320,gaydosik2023,HC,healthy_skin,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
D14__SC405,gaydosik2023,HC,healthy_skin,149,0.0087,0.0435,0,0.0174,0.0609,0.0087,0.0522,0,0.0087,0.113,0.0348,0.191,0.104,0.0609,0.0696,0.104,0.0348,0.0261,0.139,0.122,0.0348,0.122,0.0696,0.0522,0,0.0087,0.0348,0.165,0.0435,0.174,0.183,0.0696,0.243
D15__NS13,il4ra2026,HC,healthy_skin,390,0.0301,0.0936,0.00669,0.00669,0.0502,0.0368,0.191,0.0903,0.0268,0.0635,0.0569,0.291,0.0736,0.104,0.0468,0.365,0.144,0.0401,0.167,0.137,0.0602,0.411,0.087,0.187,0.00669,0.0201,0.117,0.472,0.0569,0.147,0.311,0.0234,0.197
D15__NS16,il4ra2026,HC,healthy_skin,91,0.0455,0.159,0.0114,0.0114,0.0795,0.0795,0.193,0.0682,0.0455,0.0795,0.0227,0.659,0.284,0.0227,0.114,0.148,0.0455,0,0.0568,0.102,0.0455,0.125,0.0909,0.0682,0,0.0114,0.114,0.125,0.0568,0.0795,0.25,0.114,0.227
D15__NS17,il4ra2026,HC,healthy_skin,352,0.0273,0.0887,0.0137,0.0239,0.058,0.041,0.208,0.0307,0.0205,0.099,0.0239,0.392,0.229,0.0819,0.0512,0.0819,0.0375,0.0171,0.164,0.14,0.099,0.181,0.0375,0.099,0,0,0.13,0.259,0.0137,0.109,0.177,0.0683,0.311
D15__NS14,il4ra2026,HC,healthy_skin,554,0.043,0.33,0.00452,0.0271,0.206,0.219,0.195,0.21,0.086,0.115,0.147,0.396,0.172,0.163,0.127,0.529,0.21,0.0973,0.394,0.312,0.0588,0.548,0.21,0.385,0.00226,0.0339,0.303,0.613,0.17,0.276,0.459,0.0724,0.484
```

## §6e Per-gene summary of the donor effects

**What:** how consistent each detail gene's effect is across patients.

**How:** over the columns of §6a: median and quartiles of per-donor log2FC; frac_up = share of donors with log2FC > 0; frac_up_1 = log2FC > 1; frac_down_1 = log2FC < -1; q = the headline paired q from §3.

**Full table on disk:** `tables/tgt49_detail_summary.csv`

```csv
gene,role,n_donors,median,q25,q75,min,max,frac_up,frac_up_1,frac_down_1,cd_name,q
CD70,shortlist,75,1.1,0.401,1.85,-0.849,3.33,0.92,0.56,0,CD70,3.09e-10
CD9,shortlist,75,1.08,-0.0065,2.4,-2.33,5.31,0.733,0.547,0.0667,,7.09e-07
CPM,shortlist,75,1.02,0.0724,1.87,-1.89,3.55,0.747,0.507,0.0533,,7.08e-08
COL6A3,shortlist,75,0.811,0.315,1.5,-1.26,3.48,0.88,0.427,0.0133,,3.28e-09
CD151,shortlist,75,0.794,0.0115,1.29,-1.11,2.61,0.76,0.44,0.0133,,9.39e-08
GRN,shortlist,75,0.778,0.363,1.12,-1.03,2.57,0.907,0.333,0.0133,,9.81e-10
TFRC,shortlist,75,0.731,0.404,1.13,-0.887,2.93,0.893,0.347,0,,1.67e-09
SLC39A6,shortlist,75,0.707,0.381,1.11,-0.293,2.66,0.947,0.307,0,,1.91e-10
SLC4A2,shortlist,75,0.698,0.35,1.17,-0.762,2.34,0.933,0.36,0,,1.91e-10
TNFRSF25,shortlist,75,0.698,-0.0826,1.21,-3.51,2.37,0.747,0.307,0.107,,9.14e-05
SLC1A4,shortlist,75,0.69,-0.0127,1.29,-2.1,2.83,0.747,0.307,0.0933,,5.23e-05
LGALS3,shortlist,75,0.663,-0.658,1.49,-3.76,2.86,0.587,0.48,0.16,,0.00115
CD82,shortlist,75,0.632,0.0455,1.06,-1.9,2.03,0.773,0.28,0.0133,,1.41e-07
SLC5A3,shortlist,75,0.617,0.141,1.06,-0.603,2.12,0.893,0.28,0,,1.81e-09
IL6R,shortlist,75,0.614,-0.158,1.08,-1.87,2.65,0.72,0.32,0.0933,,2.31e-05
SYPL1,shortlist,75,0.599,0.199,0.902,-1.19,2.6,0.907,0.227,0.0133,,1.15e-09
CALU,shortlist,75,0.587,0.295,0.993,-0.537,2.06,0.88,0.24,0,,6.18e-10
QSOX1,shortlist,75,0.584,0.234,0.839,-0.938,1.98,0.853,0.147,0,,7.51e-09
VDAC1,shortlist,75,0.584,0.305,0.923,-0.488,1.72,0.947,0.147,0,,1.91e-10
TNFRSF4,shortlist,75,0.542,-0.694,1.15,-3.68,2.27,0.667,0.307,0.173,,0.00308
TSPAN2,shortlist,75,0.53,0.0497,1,-1.56,2.48,0.747,0.267,0.0533,,1.24e-05
GPI,shortlist,75,0.507,0.144,0.761,-0.947,1.67,0.827,0.08,0,,2.79e-07
TMEM30A,shortlist,75,0.506,0.0808,0.69,-0.481,1.58,0.787,0.0933,0,,1.08e-08
CCR4,positive_control,75,0.324,-0.885,1.16,-3.47,2.95,0.56,0.293,0.227,CCR4,0.0909
TNFRSF8,positive_control,75,0.332,-0.112,0.964,-1.22,2.96,0.667,0.24,0.0133,CD30,0.000215
KIR3DL2,positive_control,75,0,0,2.27,-3.08,3.54,0.36,0.347,0.04,CD158k,0.000947
IL2RA,positive_control,75,-0.172,-0.753,0.384,-2.57,1.84,0.387,0.0933,0.187,CD25,0.454
CD7,loss_marker,75,-2.85,-3.84,-1.31,-5.72,0.898,0.0933,0,0.773,CD7,3.6e-09
DPP4,loss_marker,75,-0.841,-1.32,0.011,-2.38,2.85,0.253,0.12,0.44,,0.00391
CD5,loss_marker,75,-0.0584,-1.53,0.653,-3.91,2.89,0.493,0.12,0.307,CD5,0.816
CD6,loss_marker,75,0.0957,-0.593,0.531,-3.77,2.4,0.56,0.12,0.173,,0.344
CD27,loss_marker,75,-0.359,-2.28,0.767,-3.74,1.86,0.467,0.147,0.413,,0.792
CD2,loss_marker,75,-0.146,-0.84,0.178,-4.8,1.88,0.453,0.0133,0.2,,0.0478
```

## §6f Eligibility

**What:** in what share of donors at least X% of the arm's CD4 cells carry the transcript. The malignant rows are the recruitment curve; the reactive and healthy rows are the T-cell off-tumour curve.

**How:** per donor, pct_expr of the arm >= threshold, averaged over donors; malignant and reactive rows use the paired donors, healthy rows all CD4_HC donors.

**Full table on disk:** `tables/tgt49_eligibility.csv`

```csv
gene,role,arm,donors,n_donors,median_pct_expr,frac_donors_ge10,frac_donors_ge25,frac_donors_ge50,frac_donors_ge75
CD70,shortlist,CD4_HC,healthy donors,21,0.024,0,0,0,0
CD70,shortlist,CD4_malignant,paired patients,75,0.215,0.813,0.427,0.0533,0
CD70,shortlist,CD4_reactive,paired patients,75,0.0507,0.187,0.0267,0,0
CD9,shortlist,CD4_HC,healthy donors,21,0.107,0.524,0.0476,0,0
CD9,shortlist,CD4_malignant,paired patients,75,0.262,0.733,0.507,0.227,0.0267
CD9,shortlist,CD4_reactive,paired patients,75,0.0563,0.307,0.0533,0,0
CPM,shortlist,CD4_HC,healthy donors,21,0.00669,0,0,0,0
CPM,shortlist,CD4_malignant,paired patients,75,0.133,0.573,0.24,0.0533,0.0133
CPM,shortlist,CD4_reactive,paired patients,75,0.0254,0.0667,0,0,0
COL6A3,shortlist,CD4_HC,healthy donors,21,0.0158,0,0,0,0
COL6A3,shortlist,CD4_malignant,paired patients,75,0.111,0.56,0.173,0.107,0.0267
COL6A3,shortlist,CD4_reactive,paired patients,75,0.0247,0.147,0.04,0.0133,0.0133
CD151,shortlist,CD4_HC,healthy donors,21,0.0612,0.19,0,0,0
CD151,shortlist,CD4_malignant,paired patients,75,0.2,0.827,0.427,0.107,0.0133
CD151,shortlist,CD4_reactive,paired patients,75,0.0864,0.387,0.0533,0,0
GRN,shortlist,CD4_HC,healthy donors,21,0.041,0.0952,0,0,0
GRN,shortlist,CD4_malignant,paired patients,75,0.212,0.867,0.453,0.107,0
GRN,shortlist,CD4_reactive,paired patients,75,0.101,0.507,0.0267,0.0133,0
TFRC,shortlist,CD4_HC,healthy donors,21,0.118,0.571,0,0,0
TFRC,shortlist,CD4_malignant,paired patients,75,0.294,0.96,0.6,0.2,0.04
TFRC,shortlist,CD4_reactive,paired patients,75,0.143,0.64,0.16,0.0133,0
SLC39A6,shortlist,CD4_HC,healthy donors,21,0.0447,0.0476,0,0,0
SLC39A6,shortlist,CD4_malignant,paired patients,75,0.187,0.92,0.293,0.0667,0
SLC39A6,shortlist,CD4_reactive,paired patients,75,0.0776,0.227,0.0133,0.0133,0
SLC4A2,shortlist,CD4_HC,healthy donors,21,0.0156,0,0,0,0
SLC4A2,shortlist,CD4_malignant,paired patients,75,0.119,0.64,0.2,0.0267,0
SLC4A2,shortlist,CD4_reactive,paired patients,75,0.0393,0.133,0.0133,0,0
TNFRSF25,shortlist,CD4_HC,healthy donors,21,0.0943,0.333,0,0,0
TNFRSF25,shortlist,CD4_malignant,paired patients,75,0.328,0.88,0.613,0.28,0.04
TNFRSF25,shortlist,CD4_reactive,paired patients,75,0.168,0.773,0.267,0.0267,0
SLC1A4,shortlist,CD4_HC,healthy donors,21,0.0239,0.0476,0,0,0
SLC1A4,shortlist,CD4_malignant,paired patients,75,0.143,0.627,0.307,0.0533,0.0133
SLC1A4,shortlist,CD4_reactive,paired patients,75,0.0581,0.293,0.0267,0.0133,0
LGALS3,shortlist,CD4_HC,healthy donors,21,0.361,0.81,0.714,0.143,0
LGALS3,shortlist,CD4_malignant,paired patients,75,0.48,0.88,0.773,0.48,0.253
LGALS3,shortlist,CD4_reactive,paired patients,75,0.285,0.96,0.573,0.133,0
CD82,shortlist,CD4_HC,healthy donors,21,0.176,0.667,0.381,0,0
CD82,shortlist,CD4_malignant,paired patients,75,0.495,1,0.853,0.493,0.0667
CD82,shortlist,CD4_reactive,paired patients,75,0.277,0.987,0.6,0.0533,0.0133
SLC5A3,shortlist,CD4_HC,healthy donors,21,0.0499,0.0952,0,0,0
SLC5A3,shortlist,CD4_malignant,paired patients,75,0.209,0.773,0.453,0.0933,0
SLC5A3,shortlist,CD4_reactive,paired patients,75,0.0876,0.427,0.147,0.0133,0
IL6R,shortlist,CD4_HC,healthy donors,21,0.0726,0.381,0,0,0
IL6R,shortlist,CD4_malignant,paired patients,75,0.234,0.773,0.467,0.0667,0
IL6R,shortlist,CD4_reactive,paired patients,75,0.105,0.507,0.0267,0,0
SYPL1,shortlist,CD4_HC,healthy donors,21,0.179,0.667,0.19,0.0476,0
SYPL1,shortlist,CD4_malignant,paired patients,75,0.4,0.96,0.693,0.253,0
SYPL1,shortlist,CD4_reactive,paired patients,75,0.198,0.867,0.227,0.0133,0
CALU,shortlist,CD4_HC,healthy donors,21,0.0392,0.19,0,0,0
CALU,shortlist,CD4_malignant,paired patients,75,0.193,0.867,0.347,0.0533,0
CALU,shortlist,CD4_reactive,paired patients,75,0.0939,0.413,0.0533,0,0
QSOX1,shortlist,CD4_HC,healthy donors,21,0.0261,0,0,0,0
QSOX1,shortlist,CD4_malignant,paired patients,75,0.138,0.693,0.213,0.0267,0
QSOX1,shortlist,CD4_reactive,paired patients,75,0.063,0.253,0.0133,0,0
VDAC1,shortlist,CD4_HC,healthy donors,21,0.175,0.762,0.286,0,0
VDAC1,shortlist,CD4_malignant,paired patients,75,0.494,0.987,0.88,0.493,0.0933
VDAC1,shortlist,CD4_reactive,paired patients,75,0.292,0.947,0.653,0.0267,0.0133
TNFRSF4,shortlist,CD4_HC,healthy donors,21,0.137,0.762,0.0476,0,0
TNFRSF4,shortlist,CD4_malignant,paired patients,75,0.39,0.893,0.6,0.253,0.0533
TNFRSF4,shortlist,CD4_reactive,paired patients,75,0.217,0.893,0.4,0.0267,0
TSPAN2,shortlist,CD4_HC,healthy donors,21,0.0377,0.0952,0,0,0
TSPAN2,shortlist,CD4_malignant,paired patients,75,0.133,0.653,0.187,0.0133,0
TSPAN2,shortlist,CD4_reactive,paired patients,75,0.0506,0.147,0.04,0,0
GPI,shortlist,CD4_HC,healthy donors,21,0.125,0.619,0.333,0.0476,0
GPI,shortlist,CD4_malignant,paired patients,75,0.482,0.973,0.827,0.453,0.08
GPI,shortlist,CD4_reactive,paired patients,75,0.301,0.933,0.613,0.133,0.0133
TMEM30A,shortlist,CD4_HC,healthy donors,21,0.0612,0.19,0,0,0
TMEM30A,shortlist,CD4_malignant,paired patients,75,0.193,0.88,0.347,0.0133,0
TMEM30A,shortlist,CD4_reactive,paired patients,75,0.1,0.52,0.0533,0.0133,0
CCR4,positive_control,CD4_HC,healthy donors,21,0.0656,0.143,0.0476,0,0
CCR4,positive_control,CD4_malignant,paired patients,75,0.212,0.693,0.427,0.173,0.0533
CCR4,positive_control,CD4_reactive,paired patients,75,0.134,0.627,0.147,0.0267,0
TNFRSF8,positive_control,CD4_HC,healthy donors,21,0,0,0,0,0
TNFRSF8,positive_control,CD4_malignant,paired patients,75,0.0374,0.187,0.0667,0.04,0
TNFRSF8,positive_control,CD4_reactive,paired patients,75,0.0109,0.0533,0,0,0
KIR3DL2,positive_control,CD4_HC,healthy donors,21,0.00256,0,0,0,0
KIR3DL2,positive_control,CD4_malignant,paired patients,75,0,0.32,0.253,0.147,0
KIR3DL2,positive_control,CD4_reactive,paired patients,75,0.00431,0.0667,0.0133,0,0
IL2RA,positive_control,CD4_HC,healthy donors,21,0.0828,0.429,0.0476,0,0
IL2RA,positive_control,CD4_malignant,paired patients,75,0.124,0.64,0.253,0.0667,0.0133
IL2RA,positive_control,CD4_reactive,paired patients,75,0.122,0.627,0.147,0.0267,0
CD7,loss_marker,CD4_HC,healthy donors,21,0.211,0.81,0.381,0.0476,0
CD7,loss_marker,CD4_malignant,paired patients,75,0.039,0.32,0.187,0.12,0.04
CD7,loss_marker,CD4_reactive,paired patients,75,0.535,0.973,0.907,0.533,0.12
DPP4,loss_marker,CD4_HC,healthy donors,21,0.0492,0.143,0,0,0
DPP4,loss_marker,CD4_malignant,paired patients,75,0.00885,0.187,0.107,0.04,0
DPP4,loss_marker,CD4_reactive,paired patients,75,0.0444,0.16,0.0667,0.0133,0
CD5,loss_marker,CD4_HC,healthy donors,21,0.104,0.571,0.0952,0,0
CD5,loss_marker,CD4_malignant,paired patients,75,0.319,0.747,0.573,0.307,0.0267
CD5,loss_marker,CD4_reactive,paired patients,75,0.255,0.947,0.507,0.0533,0
CD6,loss_marker,CD4_HC,healthy donors,21,0.183,0.714,0.238,0,0
CD6,loss_marker,CD4_malignant,paired patients,75,0.395,0.893,0.68,0.427,0.0933
CD6,loss_marker,CD4_reactive,paired patients,75,0.345,0.947,0.667,0.12,0
CD27,loss_marker,CD4_HC,healthy donors,21,0.0816,0.429,0,0,0
CD27,loss_marker,CD4_malignant,paired patients,75,0.269,0.587,0.52,0.373,0.0267
CD27,loss_marker,CD4_reactive,paired patients,75,0.279,0.933,0.56,0.0267,0
CD2,loss_marker,CD4_HC,healthy donors,21,0.392,0.81,0.667,0.0476,0
CD2,loss_marker,CD4_malignant,paired patients,75,0.602,0.893,0.84,0.653,0.267
CD2,loss_marker,CD4_reactive,paired patients,75,0.582,1,0.973,0.707,0.107
```

## §7 lesion: strata

**What:** paired donors per lesion stratum and their malignant burden.

**How:** burden = n_malignant / (n_malignant + n_reactive) CD4 cells; mwu_p = two-sided Mann-Whitney of burden between the two strata. A small p means this split and burden are partly the same comparison.

```csv
stratum,n_paired_donors,mean,std,min,25%,50%,75%,max,mwu_p_burden_between_strata
patch_plaque,19,0.544,0.234,0.153,0.329,0.566,0.752,0.883,0.394
tumour_eryth,30,0.589,0.291,0.00837,0.359,0.624,0.827,0.985,0.394
```

## §7 lesion: detail genes per stratum

**What:** the malignant-vs-reactive effect of each detail gene inside each lesion stratum.

**How:** strat_selectivity: paired Wilcoxon inside the stratum (p only if >= 3 donors; q = BH over all gene x stratum rows); delta_lfc / q_interaction = Mann-Whitney of per-donor log2FC between the two strata, BH over genes; log2fc_all / q_all = the §3 result on all paired donors.

**Full table on disk:** `tables/tgt49_strat_lesion.csv`

```csv
gene,frac_donors_up_tumour_eryth,log2fc_paired_tumour_eryth,n_donors_paired_tumour_eryth,p_tumour_eryth,q_tumour_eryth,frac_donors_up_patch_plaque,log2fc_paired_patch_plaque,n_donors_paired_patch_plaque,p_patch_plaque,q_patch_plaque,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
CD70,0.933,1.1,30,1.86e-08,1.01e-05,0.947,1.09,19,2.67e-05,0.000952,tumour_eryth vs patch_plaque,30,19,0.0132,0.992,1,surface,1.1,3.09e-10,True
CD9,0.7,0.946,30,0.00511,0.034,0.632,1.08,19,0.0258,0.0988,tumour_eryth vs patch_plaque,30,19,-0.139,0.845,1,surface,1.08,7.09e-07,True
CPM,0.833,1.39,30,1.99e-06,0.000248,0.632,1.39,19,0.00945,0.05,tumour_eryth vs patch_plaque,30,19,0.00716,0.829,1,surface,1.02,7.08e-08,True
COL6A3,0.833,0.842,30,3.79e-06,0.000302,1,1.17,19,3.81e-06,0.000302,tumour_eryth vs patch_plaque,30,19,-0.328,0.3,0.898,surface,0.811,3.28e-09,True
CD151,0.533,0.103,30,0.0234,0.092,0.842,0.988,19,0.000164,0.00333,tumour_eryth vs patch_plaque,30,19,-0.885,0.0726,0.74,surface,0.794,9.39e-08,True
GRN,0.867,0.623,30,1.99e-06,0.000248,0.947,0.953,19,0.000164,0.00333,tumour_eryth vs patch_plaque,30,19,-0.33,0.2,0.88,surface,0.778,9.81e-10,True
TFRC,0.867,0.693,30,7.99e-06,0.00042,0.947,1.07,19,7.63e-06,0.00042,tumour_eryth vs patch_plaque,30,19,-0.38,0.0454,0.704,surface,0.731,1.67e-09,True
SLC39A6,0.933,0.512,30,2.05e-07,5.25e-05,0.947,0.84,19,1.14e-05,0.000548,tumour_eryth vs patch_plaque,30,19,-0.328,0.0354,0.678,surface,0.707,1.91e-10,True
SLC4A2,0.833,0.537,30,3.79e-06,0.000302,1,0.763,19,3.81e-06,0.000302,tumour_eryth vs patch_plaque,30,19,-0.226,0.179,0.876,surface,0.698,1.91e-10,True
TNFRSF25,0.8,0.722,30,0.000667,0.00842,0.789,0.947,19,0.000965,0.0106,tumour_eryth vs patch_plaque,30,19,-0.224,0.659,0.999,surface,0.698,9.14e-05,True
SLC1A4,0.733,0.724,30,0.0106,0.0541,0.632,0.307,19,0.169,0.35,tumour_eryth vs patch_plaque,30,19,0.416,0.845,1,surface,0.69,5.23e-05,True
LGALS3,0.5,0.0216,30,0.416,0.584,0.526,1.06,19,0.0955,0.241,tumour_eryth vs patch_plaque,30,19,-1.04,0.813,1,surface,0.663,0.00115,True
CD82,0.8,0.593,30,0.000505,0.00703,0.737,0.609,19,0.00079,0.00935,tumour_eryth vs patch_plaque,30,19,-0.0161,0.813,1,surface,0.632,1.41e-07,True
SLC5A3,0.833,0.514,30,3.05e-05,0.00105,0.947,0.846,19,1.91e-05,0.000735,tumour_eryth vs patch_plaque,30,19,-0.332,0.103,0.785,surface,0.617,1.81e-09,True
IL6R,0.767,0.844,30,0.000123,0.00274,0.789,0.687,19,0.0024,0.0198,tumour_eryth vs patch_plaque,30,19,0.157,0.43,0.933,surface,0.614,2.31e-05,True
SYPL1,0.9,0.604,30,2.55e-07,6.18e-05,0.895,0.378,19,0.0024,0.0198,tumour_eryth vs patch_plaque,30,19,0.226,0.492,0.949,surface,0.599,1.15e-09,True
CALU,0.867,0.493,30,6.92e-06,0.000413,0.789,0.587,19,0.00021,0.00372,tumour_eryth vs patch_plaque,30,19,-0.0936,0.291,0.897,surface,0.587,6.18e-10,True
QSOX1,0.833,0.329,30,2.69e-05,0.000952,0.789,0.793,19,0.0024,0.0198,tumour_eryth vs patch_plaque,30,19,-0.464,0.0905,0.777,surface,0.584,7.51e-09,True
VDAC1,0.967,0.51,30,9.31e-09,8.11e-06,0.842,0.536,19,0.000336,0.00522,tumour_eryth vs patch_plaque,30,19,-0.0253,0.959,1,surface,0.584,1.91e-10,True
TNFRSF4,0.667,0.45,30,0.0364,0.125,0.842,0.674,19,0.0141,0.0657,tumour_eryth vs patch_plaque,30,19,-0.223,0.587,0.989,surface,0.542,0.00308,True
TSPAN2,0.833,0.623,30,6.92e-06,0.000413,0.737,0.53,19,0.0141,0.0657,tumour_eryth vs patch_plaque,30,19,0.0928,0.782,1,surface,0.53,1.24e-05,True
GPI,0.767,0.49,30,0.00017,0.00333,0.789,0.595,19,0.016,0.0715,tumour_eryth vs patch_plaque,30,19,-0.105,0.798,1,surface,0.507,2.79e-07,True
TMEM30A,0.733,0.237,30,0.00017,0.00333,0.895,0.694,19,0.000164,0.00333,tumour_eryth vs patch_plaque,30,19,-0.457,0.00616,0.497,surface,0.506,1.08e-08,True
CCR4,0.767,0.808,30,0.000189,0.00356,0.526,0.314,19,0.21,0.394,tumour_eryth vs patch_plaque,30,19,0.494,0.0604,0.737,surface,0.324,0.0909,True
TNFRSF8,0.7,0.408,30,0.00761,0.0432,0.579,0.124,19,0.104,0.255,tumour_eryth vs patch_plaque,30,19,0.284,0.644,0.999,surface,0.332,0.000215,True
KIR3DL2,0.567,1.36,30,0.00223,0.0193,0.474,0,19,0.0494,0.153,tumour_eryth vs patch_plaque,30,19,1.36,0.813,1,surface,0,0.000947,True
IL2RA,0.433,-0.0735,30,0.7,0.795,0.316,-0.427,19,0.145,0.312,tumour_eryth vs patch_plaque,30,19,0.353,0.0905,0.777,surface,-0.172,0.454,True
CD7,0.1,-3.03,30,1.6e-05,0.000669,0.105,-2.76,19,0.00117,0.012,tumour_eryth vs patch_plaque,30,19,-0.27,0.943,1,surface,-2.85,3.6e-09,True
DPP4,0.133,-1.02,30,6.29e-05,0.00177,0.316,-0.796,19,0.441,0.606,tumour_eryth vs patch_plaque,30,19,-0.223,0.103,0.785,surface,-0.841,0.00391,True
CD5,0.6,0.151,30,0.237,0.431,0.579,0.0866,19,0.651,0.764,tumour_eryth vs patch_plaque,30,19,0.064,0.674,1,surface,-0.0584,0.816,True
CD6,0.6,0.115,30,0.262,0.459,0.684,0.23,19,0.023,0.0911,tumour_eryth vs patch_plaque,30,19,-0.115,0.33,0.912,surface,0.0957,0.344,True
CD27,0.7,0.603,30,0.00146,0.014,0.421,-0.364,19,0.709,0.802,tumour_eryth vs patch_plaque,30,19,0.968,0.0454,0.704,surface,-0.359,0.792,True
CD2,0.333,-0.309,30,0.00123,0.0125,0.474,-0.0295,19,0.395,0.563,tumour_eryth vs patch_plaque,30,19,-0.28,0.505,0.952,surface,-0.146,0.0478,True
```

## §7 lesion: surface genes whose effect differs between strata (q_interaction < 0.05)

**What:** every surface gene whose malignant-vs-reactive effect depends on lesion.

**How:** Mann-Whitney of per-donor log2FC between the two strata, BH over genes; sorted by delta_lfc (first stratum in `strata` minus second).

**Full table on disk:** `tables/tgt49_strat_lesion.csv`

```csv
gene,frac_donors_up_tumour_eryth,log2fc_paired_tumour_eryth,n_donors_paired_tumour_eryth,p_tumour_eryth,q_tumour_eryth,frac_donors_up_patch_plaque,log2fc_paired_patch_plaque,n_donors_paired_patch_plaque,p_patch_plaque,q_patch_plaque,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
```

## §7 stage: strata

**What:** paired donors per stage stratum and their malignant burden.

**How:** burden = n_malignant / (n_malignant + n_reactive) CD4 cells; mwu_p = two-sided Mann-Whitney of burden between the two strata. A small p means this split and burden are partly the same comparison.

```csv
stratum,n_paired_donors,mean,std,min,25%,50%,75%,max,mwu_p_burden_between_strata
advanced,55,0.623,0.269,0.00837,0.364,0.674,0.832,0.985,0.1
early,18,0.496,0.279,0.0869,0.279,0.468,0.744,0.954,0.1
```

## §7 stage: detail genes per stratum

**What:** the malignant-vs-reactive effect of each detail gene inside each stage stratum.

**How:** strat_selectivity: paired Wilcoxon inside the stratum (p only if >= 3 donors; q = BH over all gene x stratum rows); delta_lfc / q_interaction = Mann-Whitney of per-donor log2FC between the two strata, BH over genes; log2fc_all / q_all = the §3 result on all paired donors.

**Full table on disk:** `tables/tgt49_strat_stage.csv`

```csv
gene,frac_donors_up_advanced,log2fc_paired_advanced,n_donors_paired_advanced,p_advanced,q_advanced,frac_donors_up_early,log2fc_paired_early,n_donors_paired_early,p_early,q_early,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
CD70,0.909,0.958,55,1.38e-09,4.02e-07,0.944,1.49,18,7.63e-05,0.000797,advanced vs early,55,18,-0.529,0.0247,0.274,surface,1.1,3.09e-10,True
CD9,0.709,1.03,55,8.36e-06,0.000141,0.833,1.34,18,0.000328,0.00249,advanced vs early,55,18,-0.313,0.303,0.757,surface,1.08,7.09e-07,True
CPM,0.745,1.2,55,4.98e-07,1.61e-05,0.722,0.684,18,0.00158,0.00875,advanced vs early,55,18,0.511,0.236,0.696,surface,1.02,7.08e-08,True
COL6A3,0.873,0.957,55,1.48e-08,1.39e-06,0.889,0.453,18,0.000328,0.00249,advanced vs early,55,18,0.504,0.173,0.622,surface,0.811,3.28e-09,True
CD151,0.727,0.794,55,1.39e-06,3.55e-05,0.944,1.06,18,0.000107,0.00106,advanced vs early,55,18,-0.266,0.247,0.707,surface,0.794,9.39e-08,True
GRN,0.891,0.821,55,3.31e-09,6.68e-07,0.944,0.715,18,0.00042,0.00298,advanced vs early,55,18,0.106,0.863,1,surface,0.778,9.81e-10,True
TFRC,0.909,0.735,55,5.78e-09,8.43e-07,0.833,0.612,18,0.000534,0.00366,advanced vs early,55,18,0.123,0.416,0.815,surface,0.731,1.67e-09,True
SLC39A6,0.945,0.702,55,4.32e-10,3.58e-07,0.944,0.843,18,7.63e-05,0.000797,advanced vs early,55,18,-0.141,0.165,0.611,surface,0.707,1.91e-10,True
SLC4A2,0.927,0.625,55,9.57e-10,3.68e-07,1,0.783,18,7.63e-06,0.000132,advanced vs early,55,18,-0.158,0.773,0.981,surface,0.698,1.91e-10,True
TNFRSF25,0.8,0.749,55,2.08e-05,0.000291,0.556,0.0497,18,0.442,0.599,advanced vs early,55,18,0.699,0.139,0.564,surface,0.698,9.14e-05,True
SLC1A4,0.709,0.42,55,0.00416,0.0193,0.889,1.63,18,0.000107,0.00106,advanced vs early,55,18,-1.21,0.00258,0.088,surface,0.69,5.23e-05,True
LGALS3,0.509,0.119,55,0.0641,0.159,0.889,1.49,18,7.63e-05,0.000797,advanced vs early,55,18,-1.37,0.00331,0.104,surface,0.663,0.00115,True
CD82,0.764,0.632,55,1.51e-06,3.75e-05,0.833,0.706,18,0.000328,0.00249,advanced vs early,55,18,-0.0734,0.604,0.904,surface,0.632,1.41e-07,True
SLC5A3,0.873,0.548,55,1.48e-08,1.39e-06,0.944,0.551,18,0.000191,0.00166,advanced vs early,55,18,-0.00272,0.964,1,surface,0.617,1.81e-09,True
IL6R,0.782,0.649,55,3.06e-06,6.99e-05,0.5,0.108,18,0.393,0.549,advanced vs early,55,18,0.541,0.0202,0.257,surface,0.614,2.31e-05,True
SYPL1,0.909,0.66,55,1.88e-09,4.43e-07,0.889,0.344,18,0.000839,0.00522,advanced vs early,55,18,0.316,0.15,0.586,surface,0.599,1.15e-09,True
CALU,0.909,0.543,55,1.61e-09,4.02e-07,0.778,0.731,18,0.000671,0.00436,advanced vs early,55,18,-0.188,0.883,1,surface,0.587,6.18e-10,True
QSOX1,0.873,0.572,55,1.71e-08,1.58e-06,0.889,0.656,18,0.000671,0.00436,advanced vs early,55,18,-0.0842,0.203,0.654,surface,0.584,7.51e-09,True
VDAC1,0.927,0.598,55,9.57e-10,3.68e-07,1,0.586,18,7.63e-06,0.000132,advanced vs early,55,18,0.012,0.985,1,surface,0.584,1.91e-10,True
TNFRSF4,0.709,0.542,55,0.000476,0.00333,0.556,0.524,18,0.551,0.7,advanced vs early,55,18,0.0179,0.578,0.897,surface,0.542,0.00308,True
TSPAN2,0.8,0.565,55,4.56e-07,1.53e-05,0.556,0.434,18,0.325,0.478,advanced vs early,55,18,0.13,0.0829,0.461,surface,0.53,1.24e-05,True
GPI,0.8,0.518,55,1.23e-06,3.21e-05,1,0.551,18,7.63e-06,0.000132,advanced vs early,55,18,-0.0331,0.903,1,surface,0.507,2.79e-07,True
TMEM30A,0.764,0.398,55,2.57e-07,9.52e-06,0.889,0.739,18,0.000145,0.00134,advanced vs early,55,18,-0.341,0.0039,0.116,surface,0.506,1.08e-08,True
CCR4,0.673,0.825,55,0.00022,0.00188,0.167,-1.03,18,0.00401,0.0186,advanced vs early,55,18,1.85,1.1e-05,0.013,surface,0.324,0.0909,True
TNFRSF8,0.582,0.251,55,0.0166,0.0573,0.889,0.774,18,0.000252,0.00206,advanced vs early,55,18,-0.523,0.0425,0.333,surface,0.332,0.000215,True
KIR3DL2,0.455,0,55,9.77e-05,0.000999,0.0556,0,18,0.5,0.653,advanced vs early,55,18,0,0.0382,0.321,surface,0,0.000947,True
IL2RA,0.4,-0.161,55,0.487,0.643,0.333,-0.284,18,0.265,0.434,advanced vs early,55,18,0.123,0.823,1,surface,-0.172,0.454,True
CD7,0.109,-2.84,55,5.4e-08,3.14e-06,0,-3.08,18,7.63e-06,0.000132,advanced vs early,55,18,0.233,0.622,0.915,surface,-2.85,3.6e-09,True
DPP4,0.218,-1.01,55,0.000196,0.0017,0.389,-0.371,18,0.932,0.963,advanced vs early,55,18,-0.637,0.0412,0.327,surface,-0.841,0.00391,True
CD5,0.6,0.191,55,0.258,0.428,0.222,-1.01,18,0.0304,0.0907,advanced vs early,55,18,1.2,0.0138,0.21,surface,-0.0584,0.816,True
CD6,0.6,0.133,55,0.101,0.221,0.5,0.0267,18,1,1,advanced vs early,55,18,0.106,0.622,0.915,surface,0.0957,0.344,True
CD27,0.564,0.331,55,0.037,0.106,0.167,-1.96,18,0.000328,0.00249,advanced vs early,55,18,2.29,0.000618,0.0404,surface,-0.359,0.792,True
CD2,0.382,-0.305,55,0.000447,0.00315,0.667,0.2,18,0.393,0.549,advanced vs early,55,18,-0.506,0.0128,0.204,surface,-0.146,0.0478,True
```

## §7 stage: surface genes whose effect differs between strata (q_interaction < 0.05)

**What:** every surface gene whose malignant-vs-reactive effect depends on stage.

**How:** Mann-Whitney of per-donor log2FC between the two strata, BH over genes; sorted by delta_lfc (first stratum in `strata` minus second).

**Full table on disk:** `tables/tgt49_strat_stage.csv`

```csv
gene,frac_donors_up_advanced,log2fc_paired_advanced,n_donors_paired_advanced,p_advanced,q_advanced,frac_donors_up_early,log2fc_paired_early,n_donors_paired_early,p_early,q_early,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
CXCR6,0.2,-1.45,55,0.000206,0.00177,0.556,0.54,18,0.108,0.233,advanced vs early,55,18,-1.99,0.000211,0.0289,surface,-1.09,0.0275,False
NTRK2,0.509,0.0166,55,0.133,0.27,0.833,1.81,18,0.000145,0.00134,advanced vs early,55,18,-1.8,0.000443,0.0351,surface,0.309,0.00112,False
FAS,0.164,-0.956,55,1.13e-06,3.08e-05,0.833,0.555,18,0.00401,0.0186,advanced vs early,55,18,-1.51,1.49e-07,0.000351,surface,-0.674,0.0272,False
FURIN,0.473,-0.0937,55,0.645,0.772,0.833,1.41,18,0.00336,0.0162,advanced vs early,55,18,-1.5,0.000246,0.0289,surface,0.0518,0.214,False
ADAM8,0.309,-0.601,55,0.00569,0.0244,0.722,0.71,18,0.0237,0.0755,advanced vs early,55,18,-1.31,0.00059,0.0404,surface,-0.397,0.336,False
PRF1,0.182,-2.13,55,4.96e-06,0.000102,0.389,-0.84,18,0.899,0.943,advanced vs early,55,18,-1.29,0.000133,0.0282,surface,-1.79,0.00013,False
SLC7A5,0.491,-0.114,55,0.893,0.94,1,1,18,7.63e-06,0.000132,advanced vs early,55,18,-1.12,5.69e-05,0.0242,surface,0.26,0.116,False
CSF1,0.164,-0.712,55,2.16e-05,0.000298,0.667,0.202,18,0.0898,0.204,advanced vs early,55,18,-0.914,8.76e-05,0.0242,surface,-0.617,0.0117,False
IL18RAP,0.218,-0.694,55,1.77e-05,0.000257,0.611,0.147,18,0.417,0.575,advanced vs early,55,18,-0.841,0.000222,0.0289,surface,-0.564,0.00136,False
TNFRSF18,0.527,0.0252,55,0.446,0.603,0.889,0.848,18,0.000252,0.00206,advanced vs early,55,18,-0.823,0.000618,0.0404,surface,0.343,0.0138,False
GCNT1,0.2,-0.668,55,0.000105,0.00106,0.5,0.0247,18,0.495,0.651,advanced vs early,55,18,-0.693,0.000711,0.043,surface,-0.536,0.00526,False
ATP2B4,0.418,-0.177,55,0.054,0.14,0.722,0.493,18,0.0056,0.0241,advanced vs early,55,18,-0.67,0.000511,0.0365,surface,-0.0292,1,False
TLN1,0.473,-0.0478,55,0.802,0.881,0.944,0.586,18,0.000145,0.00134,advanced vs early,55,18,-0.634,0.000156,0.0282,surface,0.104,0.103,False
SLAMF7,0.109,-0.817,55,9.44e-08,4.44e-06,0.333,-0.19,18,0.702,0.809,advanced vs early,55,18,-0.627,0.000211,0.0289,surface,-0.699,7.12e-06,False
APLP2,0.6,0.14,55,0.00831,0.0332,0.944,0.746,18,3.81e-05,0.000465,advanced vs early,55,18,-0.605,0.000511,0.0365,surface,0.276,9.92e-05,False
BST2,0.545,0.0459,55,0.152,0.297,0.944,0.649,18,7.63e-05,0.000797,advanced vs early,55,18,-0.603,0.000648,0.0412,surface,0.329,0.00416,False
TMIGD2,0.0545,-0.791,55,4.36e-08,3.05e-06,0.389,-0.206,18,0.523,0.672,advanced vs early,55,18,-0.585,0.000331,0.0331,surface,-0.613,8.19e-05,False
ITGAV,0.545,0.0118,55,0.766,0.857,0.889,0.434,18,0.00042,0.00298,advanced vs early,55,18,-0.422,0.000711,0.043,surface,0.0719,0.0902,False
CCR1,0.218,-0.294,55,0.0304,0.0907,0.611,0.0665,18,0.13,0.264,advanced vs early,55,18,-0.36,0.000896,0.049,surface,-0.117,0.316,False
PTPRS,0.545,0.00718,55,0.00103,0.00619,0.167,-0.0364,18,0.0199,0.0662,advanced vs early,55,18,0.0436,0.000365,0.0331,surface,0,0.146,False
ACKR3,0.491,0,55,0.00635,0.0268,0.0556,-0.0525,18,0.00377,0.0179,advanced vs early,55,18,0.0525,0.000153,0.0282,surface,0,0.416,False
FGFR2,0.691,0.224,55,6.12e-07,1.94e-05,0.278,-0.00241,18,0.433,0.591,advanced vs early,55,18,0.227,0.000448,0.0351,surface,0.0969,1.16e-05,False
CRLF2,0.691,0.156,55,2.91e-05,0.000377,0.222,-0.148,18,0.102,0.225,advanced vs early,55,18,0.303,0.000339,0.0331,surface,0.0645,0.004,False
CACNA1I,0.4,0,55,1,1,0.0556,-0.353,18,0.000421,0.00299,advanced vs early,55,18,0.353,0.000219,0.0289,surface,-0.0576,0.102,False
PTGDR2,0.455,0,55,0.455,0.614,0.0556,-0.358,18,0.00717,0.0295,advanced vs early,55,18,0.358,0.000134,0.0282,surface,0,0.25,False
LRP6,0.691,0.27,55,9.63e-07,2.73e-05,0.222,-0.1,18,0.0557,0.143,advanced vs early,55,18,0.371,9.17e-05,0.0242,surface,0.108,0.000218,False
QSOX2,0.909,0.614,55,3.86e-09,6.8e-07,0.667,0.169,18,0.265,0.434,advanced vs early,55,18,0.445,0.000443,0.0351,surface,0.512,7.08e-08,False
CD79B,0.491,-0.0213,55,0.175,0.328,0.111,-0.917,18,0.000107,0.00106,advanced vs early,55,18,0.895,0.000818,0.0481,surface,-0.439,0.92,False
CD200R1,0.527,0.0734,55,0.675,0.789,0.0556,-0.864,18,0.00129,0.00738,advanced vs early,55,18,0.937,0.000331,0.0331,surface,-0.325,0.344,False
ICAM2,0.4,-0.348,55,0.0109,0.0408,0.111,-1.3,18,0.00042,0.00298,advanced vs early,55,18,0.956,0.000365,0.0331,surface,-0.536,0.000315,False
PTH2R,0.745,0.987,55,6.78e-08,3.4e-06,0.389,0,18,0.551,0.7,advanced vs early,55,18,0.987,0.000873,0.049,surface,0.53,2.53e-07,False
S1PR1,0.4,-0.489,55,0.00895,0.0352,0,-1.82,18,7.63e-06,0.000132,advanced vs early,55,18,1.34,7.07e-05,0.0242,surface,-0.934,0.00027,False
CCR4,0.673,0.825,55,0.00022,0.00188,0.167,-1.03,18,0.00401,0.0186,advanced vs early,55,18,1.85,1.1e-05,0.013,surface,0.324,0.0909,True
CCR7,0.6,0.505,55,0.0151,0.0532,0.0556,-1.69,18,1.53e-05,0.000231,advanced vs early,55,18,2.2,1.76e-05,0.0138,surface,-0.0856,0.332,False
CD27,0.564,0.331,55,0.037,0.106,0.167,-1.96,18,0.000328,0.00249,advanced vs early,55,18,2.29,0.000618,0.0404,surface,-0.359,0.792,True
```

## §7 disease: strata

**What:** paired donors per disease stratum and their malignant burden.

**How:** burden = n_malignant / (n_malignant + n_reactive) CD4 cells; mwu_p = two-sided Mann-Whitney of burden between the two strata. A small p means this split and burden are partly the same comparison.

```csv
stratum,n_paired_donors,mean,std,min,25%,50%,75%,max,mwu_p_burden_between_strata
MF,67,0.595,0.281,0.00837,0.335,0.658,0.83,0.985,0.732
SS,6,0.658,0.208,0.33,0.533,0.739,0.815,0.836,0.732
```

## §7 disease: detail genes per stratum

**What:** the malignant-vs-reactive effect of each detail gene inside each disease stratum.

**How:** strat_selectivity: paired Wilcoxon inside the stratum (p only if >= 3 donors; q = BH over all gene x stratum rows); delta_lfc / q_interaction = Mann-Whitney of per-donor log2FC between the two strata, BH over genes; log2fc_all / q_all = the §3 result on all paired donors.

**Full table on disk:** `tables/tgt49_strat_disease.csv`

```csv
gene,frac_donors_up_MF,log2fc_paired_MF,n_donors_paired_MF,p_MF,q_MF,frac_donors_up_SS,log2fc_paired_SS,n_donors_paired_SS,p_SS,q_SS,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
CD70,0.925,1.12,67,2.05e-11,8.22e-09,0.833,0.997,6,0.0625,0.188,MF vs SS,67,6,0.119,0.617,0.898,surface,1.1,3.09e-10,True
CD9,0.776,1.31,67,3.38e-09,1.81e-07,0.5,-0.194,6,0.562,0.783,MF vs SS,67,6,1.5,0.00583,0.108,surface,1.08,7.09e-07,True
CPM,0.716,0.951,67,1.35e-07,3.57e-06,1,1.86,6,0.0312,0.114,MF vs SS,67,6,-0.908,0.149,0.497,surface,1.02,7.08e-08,True
COL6A3,0.881,0.957,67,6.49e-10,4.86e-08,0.833,0.842,6,0.0625,0.188,MF vs SS,67,6,0.115,0.617,0.898,surface,0.811,3.28e-09,True
CD151,0.836,0.988,67,3.72e-10,3.42e-08,0.167,-0.147,6,0.0625,0.188,MF vs SS,67,6,1.13,0.000338,0.0409,surface,0.794,9.39e-08,True
GRN,0.91,0.821,67,8.93e-11,1.79e-08,1,0.474,6,0.0312,0.114,MF vs SS,67,6,0.347,0.381,0.759,surface,0.778,9.81e-10,True
TFRC,0.881,0.813,67,2.81e-10,2.88e-08,1,0.662,6,0.0312,0.114,MF vs SS,67,6,0.151,0.563,0.87,surface,0.731,1.67e-09,True
SLC39A6,0.955,0.744,67,7.92e-12,4.99e-09,0.833,0.358,6,0.0938,0.26,MF vs SS,67,6,0.386,0.0823,0.386,surface,0.707,1.91e-10,True
SLC4A2,0.97,0.763,67,6.09e-12,4.68e-09,0.667,0.578,6,0.156,0.366,MF vs SS,67,6,0.185,0.223,0.601,surface,0.698,1.91e-10,True
TNFRSF25,0.716,0.675,67,0.000161,0.00152,1,1.03,6,0.0312,0.114,MF vs SS,67,6,-0.354,0.0787,0.378,surface,0.698,9.14e-05,True
SLC1A4,0.761,0.73,67,5.76e-06,8.99e-05,0.5,-0.258,6,0.844,1,MF vs SS,67,6,0.988,0.0437,0.295,surface,0.69,5.23e-05,True
LGALS3,0.657,1.06,67,1e-05,0.000146,0,-0.572,6,0.0312,0.114,MF vs SS,67,6,1.64,0.0127,0.157,surface,0.663,0.00115,True
CD82,0.791,0.649,67,2.18e-08,7.64e-07,0.667,0.36,6,0.312,0.566,MF vs SS,67,6,0.29,0.18,0.549,surface,0.632,1.41e-07,True
SLC5A3,0.896,0.628,67,2.7e-10,2.88e-08,0.833,0.776,6,0.0625,0.188,MF vs SS,67,6,-0.149,0.822,0.972,surface,0.617,1.81e-09,True
IL6R,0.687,0.613,67,3.64e-05,0.000437,1,0.95,6,0.0312,0.114,MF vs SS,67,6,-0.337,0.359,0.742,surface,0.614,2.31e-05,True
SYPL1,0.896,0.599,67,1.66e-10,2.44e-08,1,0.181,6,0.0312,0.114,MF vs SS,67,6,0.418,0.392,0.764,surface,0.599,1.15e-09,True
CALU,0.896,0.601,67,2.76e-11,1.01e-08,0.833,0.381,6,0.0625,0.188,MF vs SS,67,6,0.219,0.498,0.847,surface,0.587,6.18e-10,True
QSOX1,0.881,0.661,67,3.57e-10,3.42e-08,0.5,-0.0214,6,1,1,MF vs SS,67,6,0.683,0.00284,0.0773,surface,0.584,7.51e-09,True
VDAC1,0.955,0.71,67,3.14e-12,4.68e-09,0.833,0.284,6,0.156,0.366,MF vs SS,67,6,0.427,0.0717,0.361,surface,0.584,1.91e-10,True
TNFRSF4,0.642,0.437,67,0.00622,0.0351,0.833,1.2,6,0.0625,0.188,MF vs SS,67,6,-0.766,0.103,0.424,surface,0.542,0.00308,True
TSPAN2,0.746,0.552,67,7.5e-06,0.000112,0.833,0.352,6,0.0625,0.188,MF vs SS,67,6,0.2,0.688,0.934,surface,0.53,1.24e-05,True
GPI,0.866,0.545,67,5.82e-09,2.73e-07,0.5,-0.105,6,1,1,MF vs SS,67,6,0.649,0.00629,0.113,surface,0.507,2.79e-07,True
TMEM30A,0.821,0.51,67,2.11e-10,2.75e-08,0.5,0.00823,6,0.688,0.888,MF vs SS,67,6,0.502,0.107,0.429,surface,0.506,1.08e-08,True
CCR4,0.507,0.0732,67,0.228,0.484,1,1.29,6,0.0312,0.114,MF vs SS,67,6,-1.21,0.0198,0.205,surface,0.324,0.0909,True
TNFRSF8,0.642,0.477,67,0.000132,0.00128,0.833,0.29,6,0.312,0.566,MF vs SS,67,6,0.187,0.603,0.894,surface,0.332,0.000215,True
KIR3DL2,0.313,0,67,0.00105,0.00779,0.833,1.33,6,0.219,0.468,MF vs SS,67,6,-1.33,0.35,0.735,surface,0,0.000947,True
IL2RA,0.388,-0.203,67,0.318,0.566,0.333,-0.115,6,0.688,0.888,MF vs SS,67,6,-0.0883,0.717,0.943,surface,-0.172,0.454,True
CD7,0.0746,-2.85,67,3.72e-10,3.42e-08,0.333,-3.25,6,0.156,0.366,MF vs SS,67,6,0.403,0.549,0.865,surface,-2.85,3.6e-09,True
DPP4,0.269,-0.826,67,0.0053,0.0307,0,-1.04,6,0.0312,0.114,MF vs SS,67,6,0.217,0.485,0.84,surface,-0.841,0.00391,True
CD5,0.478,-0.144,67,0.52,0.754,0.5,-0.0227,6,0.688,0.888,MF vs SS,67,6,-0.122,0.349,0.734,surface,-0.0584,0.816,True
CD6,0.582,0.0994,67,0.179,0.408,0.167,-0.213,6,0.438,0.709,MF vs SS,67,6,0.312,0.392,0.764,surface,0.0957,0.344,True
CD27,0.418,-0.519,67,0.827,1,0.833,0.76,6,0.0625,0.188,MF vs SS,67,6,-1.28,0.0316,0.255,surface,-0.359,0.792,True
CD2,0.478,-0.0295,67,0.0636,0.19,0.167,-0.427,6,0.0938,0.26,MF vs SS,67,6,0.398,0.264,0.645,surface,-0.146,0.0478,True
```

## §7 disease: surface genes whose effect differs between strata (q_interaction < 0.05)

**What:** every surface gene whose malignant-vs-reactive effect depends on disease.

**How:** Mann-Whitney of per-donor log2FC between the two strata, BH over genes; sorted by delta_lfc (first stratum in `strata` minus second).

**Full table on disk:** `tables/tgt49_strat_disease.csv`

```csv
gene,frac_donors_up_MF,log2fc_paired_MF,n_donors_paired_MF,p_MF,q_MF,frac_donors_up_SS,log2fc_paired_SS,n_donors_paired_SS,p_SS,q_SS,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
CCR7,0.418,-0.278,67,0.881,1,1,2.35,6,0.0312,0.114,MF vs SS,67,6,-2.63,3.16e-05,0.0149,surface,-0.0856,0.332,False
S1PR1,0.254,-1.29,67,6.41e-07,1.4e-05,0.833,1.15,6,0.156,0.366,MF vs SS,67,6,-2.44,0.000178,0.0399,surface,-0.934,0.00027,False
GPR155,0.269,-0.371,67,0.00634,0.0356,0.833,1.12,6,0.0625,0.188,MF vs SS,67,6,-1.5,0.000482,0.0409,surface,-0.322,0.274,False
CTLA4,0.507,0.086,67,0.94,1,1,1.15,6,0.0312,0.114,MF vs SS,67,6,-1.07,0.00125,0.046,surface,0.159,0.402,False
CD55,0.179,-0.734,67,5.3e-08,1.62e-06,0.833,0.193,6,0.156,0.366,MF vs SS,67,6,-0.927,0.000833,0.0409,surface,-0.65,7.84e-06,False
MFSD8,0.597,0.0673,67,0.406,0.689,1,0.552,6,0.0312,0.114,MF vs SS,67,6,-0.484,0.000429,0.0409,surface,0.0743,0.155,False
GALR2,0.388,0,67,0.815,0.998,1,0.141,6,0.0312,0.114,MF vs SS,67,6,-0.141,0.00119,0.0453,surface,0,0.379,False
KIAA0922,0.0299,0,67,0.285,0.55,0.5,0.044,6,0.25,0.494,MF vs SS,67,6,-0.044,6.76e-05,0.0227,surface,0,0.145,False
GFRA4,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2A14,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2B2,0.0149,0,67,0.317,0.566,0.333,0,6,0.5,0.729,MF vs SS,67,6,0,0.000209,0.0409,surface,0,0.195,False
OR2C1,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
HTR3B,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
MC2R,0,0,67,,,0.333,0,6,0.5,0.729,MF vs SS,67,6,0,2.32e-06,0.00182,surface,0,0.29,False
ADAM2,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
APOB,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2C3,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2H1,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2L5,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2L8,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR1L3,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR6K3,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR6F1,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR5B3,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR52B6,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR52N5,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR52N2,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR52N1,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR4B1,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR51I2,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR2T10,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
AMBP,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
SLC22A8,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
PVRL2,0.0149,0,67,0.655,0.87,0,0,6,0.5,0.729,MF vs SS,67,6,0,0.000708,0.0409,surface,0,0.592,False
RXFP2,0,0,67,,,0.333,0,6,0.5,0.729,MF vs SS,67,6,0,2.32e-06,0.00182,surface,0,0.29,False
RGR,0.0149,0,67,0.655,0.87,0.333,0,6,0.5,0.729,MF vs SS,67,6,0,0.000708,0.0409,surface,0,0.592,False
SSTR4,0,0,67,,,0,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
SLITRK3,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
PRLHR,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR56A3,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
OR9A2,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
ORM1,0.0149,0,67,0.655,0.87,0.333,0,6,0.5,0.729,MF vs SS,67,6,0,0.000708,0.0409,surface,0,0.816,False
OR8A1,0,0,67,,,0.167,0,6,1,1,MF vs SS,67,6,0,0.000996,0.0409,surface,0,0.441,False
PVRL3,0.0299,0,67,0.18,0.408,0,-0.0138,6,0.25,0.494,MF vs SS,67,6,0.0138,2.31e-06,0.00182,surface,0,0.955,False
SLC7A4,0.119,0,67,0.361,0.63,0,-0.115,6,0.0625,0.188,MF vs SS,67,6,0.115,0.00107,0.0419,surface,0,0.193,False
PTTG1IP,0.552,0.0596,67,0.0463,0.159,0,-0.34,6,0.0312,0.114,MF vs SS,67,6,0.399,0.00138,0.0491,surface,0.0136,0.29,False
ATP6AP1,0.776,0.321,67,7.29e-07,1.55e-05,0,-0.2,6,0.0312,0.114,MF vs SS,67,6,0.521,0.00125,0.046,surface,0.284,3.5e-05,False
MSN,0.716,0.21,67,7.1e-05,0.000765,0,-0.348,6,0.0312,0.114,MF vs SS,67,6,0.559,0.000263,0.0409,surface,0.173,0.00378,False
CLPTM1,0.896,0.443,67,1.46e-10,2.39e-08,0.333,-0.118,6,0.562,0.783,MF vs SS,67,6,0.561,0.000482,0.0409,surface,0.427,1.92e-08,False
TRPV2,0.433,-0.0724,67,0.216,0.468,0,-0.795,6,0.0312,0.114,MF vs SS,67,6,0.723,0.00102,0.0409,surface,-0.0853,0.0716,False
BSG,0.836,0.428,67,5.82e-09,2.73e-07,0,-0.321,6,0.0312,0.114,MF vs SS,67,6,0.749,8.62e-05,0.0254,surface,0.392,2.21e-06,False
TLN1,0.642,0.17,67,0.00191,0.0128,0,-0.672,6,0.0312,0.114,MF vs SS,67,6,0.842,0.000381,0.0409,surface,0.104,0.103,False
SLC1A5,0.746,0.507,67,2.48e-06,4.36e-05,0,-0.412,6,0.0312,0.114,MF vs SS,67,6,0.918,0.00054,0.0409,surface,0.35,0.000733,False
EMB,0.388,-0.159,67,0.0528,0.177,0,-1.1,6,0.0312,0.114,MF vs SS,67,6,0.945,0.00138,0.0491,surface,-0.2,0.0204,False
NEO1,0.851,0.561,67,3.44e-09,1.81e-07,0.333,-0.475,6,0.312,0.566,MF vs SS,67,6,1.04,0.000925,0.0409,surface,0.492,8.51e-07,False
IFNGR1,0.403,-0.0848,67,0.0417,0.146,0,-1.12,6,0.0312,0.114,MF vs SS,67,6,1.04,0.000925,0.0409,surface,-0.21,0.00904,False
CD151,0.836,0.988,67,3.72e-10,3.42e-08,0.167,-0.147,6,0.0625,0.188,MF vs SS,67,6,1.13,0.000338,0.0409,surface,0.794,9.39e-08,True
ITGAE,0.284,0,67,0.712,0.914,0,-1.19,6,0.0312,0.114,MF vs SS,67,6,1.19,0.000186,0.0399,surface,0,0.143,False
CTSD,0.493,-0.01,67,0.861,1,0,-1.45,6,0.0312,0.114,MF vs SS,67,6,1.44,5.34e-05,0.0209,surface,-0.127,0.543,False
CD63,0.672,0.25,67,0.00225,0.0148,0,-1.28,6,0.0312,0.114,MF vs SS,67,6,1.53,0.0001,0.0263,surface,0.208,0.115,False
```

======================== COPY TO HERE ==========================