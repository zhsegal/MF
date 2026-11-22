# 51 - Payload-selectivity screen (Gate 1-3), CD70 TRUCK, MF/CTCL skin

_Written by `jobs/run_payload_screen.py` on 2026-11-19 16:42._

Every table below is a CSV block with a header row, printed at 3 significant figures. `q` is BH-adjusted unless the **How** line says otherwise. The on-disk table backing each block is named above it.

## Context

**Object.** T/NK arms from `skin_T_tcr_annotated_v4.h5ad` (data/atlas_joint/skin_T_tcr_annotated_v4.h5ad), layer `raw_counts`;
myeloid, fibroblast and context arms from `joint_annotated.h5ad`, whose `var` is the same
42,347-gene index, with subtype labels joined from `skin_myeloid_fibro_subtypes.csv`.

**Malignancy call.** `mal_cluster` -- the project headline call. CD4 cells split into
`CD4_malignant` / `CD4_reactive` / `CD4_unassessed` by `malignancy.cd4_level`; `CD4_HC` is
then set from the curated diagnosis (`clin_dx`), not from the atlas `disease` column.
`CD4_unassessed` is not used here.

**Normalisation.** log1p(counts / obs.total_counts * 1e4), per cell. The denominator is the true 42k-gene library
`obs.total_counts`, not the sum over the columns read.

**Arms.** CD4_malignant, CD4_reactive, CD4_HC, CD8, CD4_Treg, NK, Mono_Mac, cDC1, cDC2, pDC, Fibroblast, Keratinocyte, Vascular. Every arm uses the same >= 20-cell floor per donor
and the same donor keys, so arms are joinable per donor. NK overlaps the CD4/CD8 arms (§1d).

**Cohort.** `target_cohort.skin_donor_table()` + `filter_long`: 146
donor keys reduce to 109 independent keys
(107 people) after dropping non-CD4 clones, diagnoses outside the MF/SS
spectrum and repeat biopsies. 75 donors carry both CD4 arms and enter
the paired test; the healthy arm has 17 donors after the empty-row rule.

**Gene families.** A permitted (17), B disqualified (13),
C chemoattractant axis (23), plus 5 audit genes. BH correction runs
**within family**, and separately from the 2,115-gene surface screen -- this is a targeted
hypothesis set, not a genome scan.

**Tests.** Paired Wilcoxon signed-rank on per-donor differences, malignant vs reactive CD4
within the same biopsy, run twice: on `mean_all` (expression) and on `pct_expr` (detection).
Donor-level only. Both are reported side by side and neither is preferred.

**What the gates can and cannot establish.**
- *Gate 1* establishes that a receptor's transcript is higher or lower on the clone than on the
  patient's own reactive CD4. It does not establish surface protein, and a depleted transcript
  is not proof the clone cannot respond to the cytokine.
- *Gate 2* establishes what fraction of cells carry every chain of a complex as detected mRNA.
  Co-detection of two sparse transcripts is a severe underestimate of the true co-expression,
  so Gate 2 is a lower bound, and a comparison between arms of different depth is confounded.
- *Gate 3* establishes whether the ligand transcript is already present in the lesion and in
  which population. It says nothing about protein, about secretion, or about the gradient a
  cell would actually see.
- None of the three establishes efficacy. They establish selectivity, reachability and
  redundancy, which is what chooses between payloads, not whether a payload works.

## §1a Run manifest

**What:** The object, the call, the normalisation and the thresholds every number below rests on.

**How:** Written by this job to `data/targets/pay51_manifest.json`; keys are literal run parameters, not derived quantities.

```csv
key,value
written,2026-11-19 16:42
builder,run_payload_screen.py
object_T,data/atlas_joint/skin_T_tcr_annotated_v4.h5ad
object_nonT,data/atlas_joint/joint_annotated.h5ad
layer,raw_counts
malignancy_call,mal_cluster
normalisation,"log1p(counts / obs.total_counts * 1e4), per cell"
min_cells_per_donor_arm,20
min_donors_per_test,5
pseudocount_log2fc,0.05
detect_floor,0.1
n_genes,58
donor_keys_in_object,146
independent_donor_keys,109
people,107
paired_donors,75
hc_donors,17
empty_count_rows_dropped_T,1083
empty_count_rows_dropped_joint,0
unmeasured_cells_masked,325
```

## §1b Arms

**What:** One row per arm: donors, cells, studies, and the sequencing depth of its cells.

**How:** Donor-arms with >= 20 cells after the empty-row rule and the cohort filter. median_total_counts and median_n_genes are the median over donor-arms of the per-donor median UMI and genes per cell, from `obs`. Depth is NOT matched between arms in Gate 1 or 2: see §6d.

**Table on disk:** `tables/pay51_arms.csv`

```csv
arm,n_donors,n_cells,n_studies,median_total_counts,median_n_genes
CD4_malignant,78,187748,12,3.33e+03,1.6e+03
CD4_reactive,82,147233,12,2.04e+03,1.07e+03
CD4_HC,17,3818,5,2.35e+03,870
CD8,84,45480,12,2.08e+03,1.04e+03
CD4_Treg,76,35534,12,2.51e+03,1.32e+03
NK,24,3344,1,1.9e+03,1.13e+03
Mono_Mac,99,31461,13,3.11e+03,1.34e+03
cDC1,52,6594,11,4.31e+03,1.54e+03
cDC2,82,14939,13,5.99e+03,1.82e+03
pDC,13,535,4,7.1e+03,1.91e+03
Fibroblast,98,120309,13,4.81e+03,1.9e+03
Keratinocyte,93,209494,13,4.58e+03,1.7e+03
Vascular,101,78599,13,3.91e+03,1.79e+03
```

The malignant arm is sequenced deeper than the reactive arm, so every DETECTION comparison (pct_expr) is depth-confounded in the direction that favours the malignant arm. §6d repeats Gate 1 detection after binomial thinning to a common 2,000 UMI.

## §1c Empty count rows

**What:** Cells that exist in the integrated object with QC metadata but hold no counts at all.

**How:** Non-zero entries per cell read from the CSR indptr of `layers/raw_counts`. A cell with zero non-zeros is dropped from every arm here. Their source cohort object has the same cells intact, so the counts were lost when the atlas was assembled, not at sequencing.

```csv
study,donor,cells_with_empty_count_row,cells_in_object
gaydosik2023,D14__SC320,140,140
gaydosik2023,D14__SC348,494,494
gaydosik2023,D14__SC353,253,253
gaydosik2023,D14__SC354,111,111
gaydosik2023,D14__SC405,85,149
```

1,083 cells dropped in the T object, 0 in the joint object. This removes four CD4_HC donors (SC320, SC348, SC353, SC354) entirely and trims SC405; the corrected healthy n is in §1a (`hc_donors`). The four were NOT repaired from the source object -- that is a second data path for one study and was left for a separate job.

## §1d NK arm overlap

**What:** Where the NK-annotated cells sit in the T arms. The NK arm is not disjoint from them.

**How:** `obs.cell_type in (NK, ILC1_NK)` crossed with the arm the same cells already carry. The CD4/CD8/Treg arms are left exactly as `run_target_screen_qc.py` defines them, so Gate 1 stays comparable with tgt49; the NK arm is additional and overlapping.

```csv
arm_of_NK_cells,li2024
CD4_malignant,9
CD4_reactive,3163
CD8,279
```

Every NK-annotated cell in this atlas comes from one study (li2024), so the NK arm is a single-cohort observation and carries no between-study replication.

## §2a Gene families

**What:** The genes tested, by family. Families are corrected separately from each other and from the 2,115-gene surface screen.

**How:** Family membership is fixed a priori, not data-driven. `audit` genes are carried for diagnostics (CD2/CD3E/PTPRC for the empty-row check, GAPDH/CD7 for the coverage check) and are never pooled into a family or into a BH correction with one.

```csv
family,n_genes,genes,absent_from_object
A_permitted,17,IL12RB1 IL12RB2 IL18R1 IL18RAP IL2RB IL2RG IL15RA IL7R IL10RA CD40 TNFRSF9 CD27 IFNGR1 IFNGR2 IL23R TNFRSF1A TNFRSF1B,
B_disqualified,13,IL21R IL4R IL6R IL27RA IL9R IL10RB IL13RA1 IL13RA2 TNFRSF25 TNFRSF4 TNFRSF8 TNFRSF18 IL2RA,
C_axis,23,XCR1 CXCR3 CCR5 CCR4 CCR7 S1PR1 CX3CR1 XCL1 XCL2 CXCL9 CXCL10 CXCL11 CCL19 CCL21 IL12A IL12B IL18 IL18BP IL15 FLT3LG CD70 TNFSF9 TNFSF4,
audit,5,CD2 CD3E PTPRC CD7 GAPDH,
```

No gene of the set is absent from the object's 42,347-gene var.

## §2b Per-study coverage

**What:** For each gene and study, the number of cohort donors whose source object actually measured that gene. 0 means the gene is not in that cohort's gene list at all.

**How:** The mask is built from the `var` index of each cohort's own source object (`data/<cohort>/processed/standardized.h5ad`, and `data/CTCL_all_final_portal_tags.h5ad` for li2024), joined on the cohort prefix of the donor key. Unmeasured (study, gene) pairs are set to NaN and dropped before any test; they never enter a donor count as a zero. The mask is then checked against the data: an unmeasured pair must be observed all-zero, and 0 of 325 contradicted it.

**Table on disk:** `tables/pay51_gene_coverage.csv`

```csv
gene,alkon2024,brentuximab2026,brunner2024,buus2025,chennareddy2025,gaydosik2019,gaydosik2022,gaydosik2023,herrera2021,il4ra2026,jonak2021,li2024,rindler2021_mc
IL12RB1,5,6,5,3,9,7,9,5,4,14,1,35,4
IL12RB2,5,6,5,3,9,7,9,5,4,14,1,35,4
IL18R1,5,6,5,3,9,7,9,5,4,14,1,35,4
IL18RAP,5,6,5,3,9,7,9,5,4,14,1,35,4
IL2RB,5,6,5,3,9,7,9,5,4,14,1,35,4
IL2RG,5,6,5,3,9,7,9,5,4,14,1,35,4
IL15RA,5,6,5,3,9,7,9,5,4,14,1,35,4
IL7R,5,6,5,3,9,7,9,5,4,14,1,35,4
IL10RA,5,6,5,3,9,7,9,5,4,14,1,35,4
CD40,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF9,5,6,5,3,9,7,9,5,4,14,1,35,4
CD27,5,6,5,3,9,7,9,5,4,14,1,35,4
IFNGR1,5,6,5,3,9,7,9,5,4,14,1,35,4
IFNGR2,5,6,5,3,9,7,9,5,4,14,1,35,4
IL23R,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF1A,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF1B,5,6,5,3,9,7,9,5,4,14,1,35,4
IL21R,5,6,5,3,9,7,9,5,4,14,1,35,4
IL4R,5,6,5,3,9,7,9,5,4,14,1,35,4
IL6R,5,6,5,3,9,7,9,5,4,14,1,35,4
IL27RA,5,6,5,3,9,7,9,5,4,14,1,35,4
IL9R,5,6,5,3,9,7,9,5,4,14,1,35,4
IL10RB,5,6,5,3,9,7,9,5,4,14,1,35,4
IL13RA1,5,6,5,3,9,7,9,5,4,14,1,35,4
IL13RA2,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF25,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF4,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF8,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFRSF18,5,6,5,3,9,7,9,5,4,14,1,35,4
IL2RA,5,6,5,3,9,7,9,5,4,14,1,35,4
XCR1,5,6,5,3,9,7,9,5,4,14,1,35,4
CXCR3,5,6,5,3,9,7,9,5,4,14,1,35,4
CCR5,5,6,5,3,9,7,9,5,4,14,1,35,4
CCR4,5,6,5,3,9,7,9,5,4,14,1,35,4
CCR7,5,6,5,3,9,7,9,5,4,14,1,35,4
S1PR1,5,6,5,3,9,7,9,5,4,14,1,35,4
CX3CR1,5,6,5,3,9,7,9,5,4,14,1,35,4
XCL1,5,6,5,3,9,7,9,5,4,14,1,35,4
XCL2,5,6,5,3,9,7,9,5,4,14,1,35,4
CXCL9,5,6,5,3,9,7,9,5,4,14,1,35,4
CXCL10,5,6,5,3,9,7,9,5,4,14,1,35,4
CXCL11,5,6,5,3,9,7,9,5,4,14,1,35,4
CCL19,5,6,5,3,9,7,9,5,4,14,1,35,4
CCL21,5,6,5,3,9,7,9,5,4,14,1,35,4
IL12A,5,6,5,3,9,7,9,5,4,14,1,35,4
IL12B,5,6,5,3,9,7,9,5,4,14,1,35,4
IL18,5,6,5,3,9,7,9,5,4,14,1,35,4
IL18BP,5,6,5,3,9,7,9,5,4,14,1,35,4
IL15,5,6,5,3,9,7,9,5,4,14,1,35,4
FLT3LG,5,6,5,3,9,7,9,5,4,14,1,35,4
CD70,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFSF9,5,6,5,3,9,7,9,5,4,14,1,35,4
TNFSF4,5,6,5,3,9,7,9,5,4,14,1,35,4
CD2,5,6,5,3,9,7,9,5,4,14,1,35,4
CD3E,5,6,5,3,9,7,9,5,4,14,1,35,4
PTPRC,5,6,5,3,9,7,9,5,4,14,1,35,4
CD7,5,6,5,3,9,7,9,5,4,14,1,35,4
GAPDH,5,6,5,3,9,7,9,5,4,14,1,0,4
```

Every gene of this payload set is measured in every cohort that contributes a skin donor, so the mask removes nothing here: 1 (study, gene) pairs are unmeasured. That is the result, not a reason the check was unnecessary -- in the 2,115-gene surface screen the same mask would remove 342 genes from li2024 alone (31 of the 75 paired donors), which is why `n_donors_paired` is a constant 75 in every tgt49 table. See §7.

## §3a Gate 1 -- permitted payload receptors

**What:** Receptors a payload is permitted to engage: the hypothesis is that they are DEPLETED on the malignant clone, so the payload acts on the reactive compartment and not on the tumour.

**How:** Paired Wilcoxon signed-rank over donors that carry both arms, on the donor-level value; donor-level only, no cell is a unit of analysis. `log2fc_paired` is the median over donors of log2((mean_all malignant + 0.05)/(mean_all reactive + 0.05)) -- a ratio of log-scale means, not a raw fold change. `log2fc_detect` is the same construction on `pct_expr` (fraction of cells with >= 1 count). BH within family over the genes of that family only. Positive = higher on the malignant clone.

**Table on disk:** `tables/pay51_cd4_paired.csv`

```csv
gene,family,n_donors_paired,log2fc_paired,q,frac_donors_up,n_up,n_down,log2fc_detect,q_detect,malig_median_pct_expr,react_median_pct_expr,hc_median_pct_expr,malig_frac_donors_positive
IFNGR2,A_permitted,75,0.294,0.00703,0.6,45,29,0.46,0.000103,0.0707,0.0403,0.0392,0.372
IL2RB,A_permitted,75,0.141,0.263,0.587,44,31,0.327,0.000249,0.32,0.264,0.14,0.91
IL2RG,A_permitted,75,0.176,0.004,0.667,50,25,0.17,1.24e-05,0.72,0.699,0.482,1
TNFRSF1A,A_permitted,75,0.0136,0.815,0.507,38,37,0.164,0.00194,0.1,0.082,0.0486,0.513
IL12RB1,A_permitted,75,-0.152,0.191,0.453,34,41,0.146,0.0312,0.113,0.103,0.05,0.551
TNFRSF1B,A_permitted,75,-0.15,0.191,0.453,34,41,0.109,0.18,0.333,0.341,0.275,0.923
IL15RA,A_permitted,75,-0.0328,0.263,0.44,33,41,0.0621,0.0434,0.0275,0.0335,0.0115,0.167
IFNGR1,A_permitted,75,-0.21,0.00585,0.36,27,48,0.0483,0.178,0.115,0.133,0.0984,0.628
IL23R,A_permitted,75,-0.0862,0.822,0.373,28,42,0,0.503,0.00446,0.00718,0.0113,0.0897
CD27,A_permitted,75,-0.359,0.78,0.467,35,40,-0.013,0.273,0.246,0.278,0.114,0.59
CD40,A_permitted,75,-0.125,0.000276,0.253,19,51,-0.0465,0.0147,0.0049,0.0075,0.00559,0.0128
IL18R1,A_permitted,75,-0.325,0.0019,0.267,20,54,-0.0793,0.473,0.052,0.0623,0.0226,0.256
TNFRSF9,A_permitted,75,-0.266,0.102,0.387,29,46,-0.111,0.996,0.0731,0.0814,0.0401,0.372
IL12RB2,A_permitted,75,-0.449,3.42e-05,0.173,13,62,-0.212,0.00657,0.015,0.036,0.00767,0.192
IL18RAP,A_permitted,75,-0.564,0.00105,0.307,23,51,-0.257,0.0728,0.00957,0.032,0.0204,0.128
IL7R,A_permitted,75,-0.775,1.52e-10,0.107,8,67,-0.392,1.38e-05,0.41,0.601,0.608,0.91
IL10RA,A_permitted,75,-0.783,3.33e-09,0.187,14,61,-0.472,9.75e-05,0.206,0.32,0.274,0.782
```

## §3b Gate 1 -- disqualified payload receptors

**What:** Receptors that disqualify a payload: the hypothesis is that they are ENRICHED on the clone, so the payload would feed it. They are kept in and reported -- that is the result.

**How:** Paired Wilcoxon signed-rank over donors that carry both arms, on the donor-level value; donor-level only, no cell is a unit of analysis. `log2fc_paired` is the median over donors of log2((mean_all malignant + 0.05)/(mean_all reactive + 0.05)) -- a ratio of log-scale means, not a raw fold change. `log2fc_detect` is the same construction on `pct_expr` (fraction of cells with >= 1 count). BH within family over the genes of that family only. Positive = higher on the malignant clone.

**Table on disk:** `tables/pay51_cd4_paired.csv`

```csv
gene,family,n_donors_paired,log2fc_paired,q,frac_donors_up,n_up,n_down,log2fc_detect,q_detect,malig_median_pct_expr,react_median_pct_expr,hc_median_pct_expr,malig_frac_donors_positive
TNFRSF25,B_disqualified,75,0.698,5.87e-05,0.747,56,19,0.821,7.66e-08,0.339,0.171,0.0984,0.872
IL6R,B_disqualified,75,0.614,1.78e-05,0.72,54,21,0.778,9.81e-09,0.233,0.107,0.102,0.769
IL10RB,B_disqualified,75,0.371,6.65e-06,0.76,57,18,0.645,4.78e-10,0.166,0.0907,0.0408,0.756
IL4R,B_disqualified,75,0.286,0.000638,0.653,49,26,0.563,5.89e-08,0.222,0.156,0.0871,0.885
IL9R,B_disqualified,75,0.472,0.000638,0.64,48,27,0.541,1.43e-05,0.0693,0.0342,0.0447,0.41
TNFRSF4,B_disqualified,75,0.542,0.0013,0.667,50,25,0.538,7.06e-06,0.349,0.223,0.14,0.897
IL21R,B_disqualified,75,0.203,0.00295,0.627,47,28,0.534,1.55e-06,0.239,0.154,0.0784,0.795
IL27RA,B_disqualified,75,0.313,0.000321,0.68,51,24,0.479,9.81e-09,0.304,0.191,0.188,0.897
TNFRSF8,B_disqualified,75,0.332,0.000116,0.667,50,25,0.455,1.7e-07,0.038,0.0111,0,0.192
TNFRSF18,B_disqualified,75,0.343,0.00595,0.627,47,28,0.442,2.08e-05,0.369,0.25,0.311,0.833
IL13RA1,B_disqualified,75,0.0132,0.129,0.533,40,28,0.0306,0.00139,0.00428,0.00127,0,0.0513
IL2RA,B_disqualified,75,-0.172,0.329,0.387,29,46,0.0171,0.141,0.122,0.126,0.104,0.628
IL13RA2,B_disqualified,75,0,0.0459,0.267,20,7,0,0.00339,0,0,0,0.0128
```

## §3c Gate 1 -- chemoattractant axis

**What:** The chemoattractant axis: receptors that decide who is recruited, and ligands that decide whether the signal is already present.

**How:** Paired Wilcoxon signed-rank over donors that carry both arms, on the donor-level value; donor-level only, no cell is a unit of analysis. `log2fc_paired` is the median over donors of log2((mean_all malignant + 0.05)/(mean_all reactive + 0.05)) -- a ratio of log-scale means, not a raw fold change. `log2fc_detect` is the same construction on `pct_expr` (fraction of cells with >= 1 count). BH within family over the genes of that family only. Positive = higher on the malignant clone.

**Table on disk:** `tables/pay51_cd4_paired.csv`

```csv
gene,family,n_donors_paired,log2fc_paired,q,frac_donors_up,n_up,n_down,log2fc_detect,q_detect,malig_median_pct_expr,react_median_pct_expr,hc_median_pct_expr,malig_frac_donors_positive
CD70,C_axis,75,1.1,2.72e-11,0.92,69,6,1.22,1.97e-12,0.21,0.0498,0.0273,0.808
CCR4,C_axis,75,0.324,0.0645,0.56,42,33,0.568,0.00138,0.219,0.135,0.0726,0.705
TNFSF9,C_axis,75,0.32,0.00377,0.613,46,29,0.485,1.17e-05,0.084,0.0591,0.061,0.449
TNFSF4,C_axis,75,0.289,0.000333,0.68,51,24,0.295,8.43e-06,0.0416,0.0153,0.0164,0.231
CCR7,C_axis,75,-0.0856,0.269,0.48,36,39,0.197,0.018,0.307,0.182,0.1,0.692
CXCL10,C_axis,75,0.0999,0.000721,0.64,48,25,0.137,7.51e-06,0.0125,0.00353,0,0.103
CCL19,C_axis,75,-0.0028,0.0225,0.387,29,38,0,0.853,0.00627,0.00629,0.00341,0.128
CCL21,C_axis,75,-0.00858,0.0225,0.253,19,40,0,0.853,0.00348,0.0028,0,0.0128
CXCL11,C_axis,75,0,0.342,0.387,29,20,0,0.0453,0.000272,0,0,0
CXCL9,C_axis,75,0,0.0728,0.36,27,34,0,0.853,0.00325,0.00131,0,0.0385
IL12A,C_axis,75,0,0.897,0.427,32,31,0,0.096,0.00104,0.00119,0,0.0128
IL12B,C_axis,75,0,0.331,0.187,14,7,0,0.165,0,0,0,0
IL15,C_axis,75,-0.0325,0.743,0.4,30,42,0,0.35,0.00661,0.00776,0,0.0256
XCR1,C_axis,75,0,0.342,0.293,22,13,0,0.156,0,0,0,0.0128
IL18BP,C_axis,75,-0.224,0.0669,0.347,26,48,-0.0609,0.853,0.0229,0.0272,0.0137,0.192
CX3CR1,C_axis,75,-0.246,0.00021,0.227,17,54,-0.112,0.00327,0.00226,0.0103,0.00341,0.0128
IL18,C_axis,75,-0.215,2.03e-08,0.12,9,60,-0.112,1.58e-07,0.00192,0.00764,0.00669,0
FLT3LG,C_axis,75,-0.618,2.38e-06,0.187,14,61,-0.21,0.0362,0.185,0.284,0.317,0.859
CCR5,C_axis,75,-0.409,0.000257,0.187,14,59,-0.279,0.00232,0.00517,0.0235,0.0125,0.0769
S1PR1,C_axis,75,-0.934,0.00021,0.32,24,51,-0.596,0.0158,0.0613,0.109,0.0332,0.436
XCL1,C_axis,75,-1.56,0.00152,0.2,15,60,-0.819,0.00805,0.0109,0.0927,0.138,0.231
CXCR3,C_axis,75,-1.24,0.0645,0.347,26,49,-0.819,0.482,0.0607,0.158,0.0755,0.436
XCL2,C_axis,75,-1.74,0.000174,0.173,13,62,-0.953,0.000407,0.00572,0.0879,0.082,0.154
```

## §3d Gate 1 -- audit genes (not a family)

**What:** Diagnostic genes. CD2/CD3E/PTPRC must be high in every T arm (they are what showed the empty-row defect); GAPDH and CD7 check the coverage mask and the tgt49 loss-marker result.

**How:** Paired Wilcoxon signed-rank over donors that carry both arms, on the donor-level value; donor-level only, no cell is a unit of analysis. `log2fc_paired` is the median over donors of log2((mean_all malignant + 0.05)/(mean_all reactive + 0.05)) -- a ratio of log-scale means, not a raw fold change. `log2fc_detect` is the same construction on `pct_expr` (fraction of cells with >= 1 count). BH within family over the genes of that family only. Positive = higher on the malignant clone.

**Table on disk:** `tables/pay51_cd4_paired.csv`

```csv
gene,family,n_donors_paired,log2fc_paired,q,frac_donors_up,n_up,n_down,log2fc_detect,q_detect,malig_median_pct_expr,react_median_pct_expr,hc_median_pct_expr,malig_frac_donors_positive
CD3E,audit,75,-0.0939,0.0266,0.373,28,47,0.0782,0.0487,0.768,0.736,0.607,0.974
CD2,audit,75,-0.146,0.0266,0.453,34,41,0.0676,0.975,0.6,0.583,0.443,0.897
PTPRC,audit,75,-0.0324,0.0559,0.4,30,45,0.0515,0.0826,0.822,0.847,0.714,1
GAPDH,audit,45,0.16,0.000121,0.756,34,11,0.0384,0.0432,0.944,0.918,0.914,1
CD7,audit,75,-2.85,3.02e-10,0.0933,7,68,-2.28,6e-11,0.038,0.539,0.259,0.308
```

## §3e Per-donor detection, every arm

**What:** One table per arm: donor x gene fraction of cells with >= 1 count, so any gene can be re-examined per patient without recomputing anything.

**How:** `pct_expr` of each (arm, donor) group, pivoted. Columns are in family order. A blank cell is a donor whose arm has fewer than 20 cells, or a (study, gene) the coverage mask marked unmeasured -- blank is never 0. `mean_all` for the same groups is in `data/targets/pay51_long.parquet`.

```csv
arm,n_donors,n_genes,median_pct_expr_over_all,file
CD4_malignant,78,58,0.0716,tables/pay51_donor_pct_CD4_malignant.csv
CD4_reactive,82,58,0.0741,tables/pay51_donor_pct_CD4_reactive.csv
CD4_HC,17,58,0.0468,tables/pay51_donor_pct_CD4_HC.csv
CD8,84,58,0.0484,tables/pay51_donor_pct_CD8.csv
CD4_Treg,76,58,0.08,tables/pay51_donor_pct_CD4_Treg.csv
NK,24,58,0.0639,tables/pay51_donor_pct_NK.csv
Mono_Mac,99,58,0.0343,tables/pay51_donor_pct_Mono_Mac.csv
cDC1,52,58,0.0345,tables/pay51_donor_pct_cDC1.csv
cDC2,82,58,0.04,tables/pay51_donor_pct_cDC2.csv
pDC,13,58,0.0698,tables/pay51_donor_pct_pDC.csv
Fibroblast,98,58,0.029,tables/pay51_donor_pct_Fibroblast.csv
Keratinocyte,93,58,0.0193,tables/pay51_donor_pct_Keratinocyte.csv
Vascular,101,58,0.0256,tables/pay51_donor_pct_Vascular.csv
```

## §4a Gate 2 -- reach, by payload and arm

**What:** The fraction of cells that carry the COMPLETE receptor complex for each payload, per arm, and how many donors reach a usable level.

**How:** Co-expression is evaluated per cell on raw counts (>= 1 count of every chain in the same cell), then averaged within each (arm, donor) group, then summarised over donors. `donor_prev_geX` is the fraction of donors in that arm whose group has >= X% reachable cells -- a donor prevalence, not a cohort mean, because a cohort mean hides whether the reach sits in a few patients. Complexes: IL-12 = IL12RB1 AND IL12RB2; IL-18 = IL18R1 AND IL18RAP; IL-15 = IL2RB AND IL2RG; XCL1 = XCR1; CXCL9/10 = CXCR3. Descriptive: no test is run here.

**Table on disk:** `tables/pay51_gate2_summary.csv`

```csv
payload,arm,n_donors,median_frac,mean_frac,max_frac,donor_prev_ge1pct,donor_prev_ge5pct,donor_prev_ge10pct
CXCL9/10,CD4_malignant,78,0.0607,0.169,0.773,0.833,0.538,0.436
CXCL9/10,CD4_reactive,82,0.158,0.189,0.627,1,0.963,0.756
CXCL9/10,CD4_HC,17,0.0755,0.0755,0.142,1,0.706,0.235
CXCL9/10,CD8,84,0.262,0.288,0.887,0.988,0.976,0.929
CXCL9/10,CD4_Treg,76,0.165,0.2,0.656,0.947,0.868,0.684
CXCL9/10,NK,24,0.12,0.171,0.457,1,0.875,0.625
CXCL9/10,Mono_Mac,99,0.00276,0.0164,0.182,0.374,0.0909,0.0303
CXCL9/10,cDC1,52,0.212,0.227,0.613,1,0.923,0.865
CXCL9/10,cDC2,82,0.0283,0.0352,0.25,0.78,0.22,0.0488
CXCL9/10,pDC,13,0.2,0.225,0.52,1,0.923,0.846
CXCL9/10,Fibroblast,98,0.00726,0.0225,0.247,0.439,0.122,0.051
CXCL9/10,Keratinocyte,93,0.00279,0.0129,0.247,0.301,0.0753,0.0215
CXCL9/10,Vascular,101,0.00427,0.0111,0.0938,0.277,0.0594,0
IL-12,CD4_malignant,78,0.00254,0.0175,0.221,0.269,0.0897,0.0641
IL-12,CD4_reactive,82,0.00645,0.016,0.127,0.439,0.0488,0.0244
IL-12,CD4_HC,17,0,0.000734,0.00559,0,0,0
IL-12,CD8,84,0,0.00575,0.0704,0.167,0.0357,0
IL-12,CD4_Treg,76,0.00756,0.0295,0.237,0.434,0.224,0.105
IL-12,NK,24,0.0101,0.0228,0.11,0.5,0.167,0.0417
IL-12,Mono_Mac,99,0,0.00216,0.0616,0.0707,0.0101,0
IL-12,cDC1,52,0,0.0016,0.0204,0.0769,0,0
IL-12,cDC2,82,0,0.000241,0.0051,0,0,0
IL-12,pDC,13,0,0.00844,0.04,0.231,0,0
IL-12,Fibroblast,98,0,0.00279,0.0674,0.0714,0.0102,0
IL-12,Keratinocyte,93,0,0.00265,0.0608,0.086,0.0108,0
IL-12,Vascular,101,0,0.00114,0.0323,0.0396,0,0
IL-15,CD4_malignant,78,0.268,0.319,0.922,0.987,0.949,0.885
IL-15,CD4_reactive,82,0.198,0.228,0.684,1,0.927,0.768
IL-15,CD4_HC,17,0.082,0.1,0.294,0.941,0.706,0.471
IL-15,CD8,84,0.164,0.207,0.746,1,0.893,0.69
IL-15,CD4_Treg,76,0.341,0.348,0.911,1,0.974,0.934
IL-15,NK,24,0.553,0.551,0.915,1,0.958,0.958
IL-15,Mono_Mac,99,0.00114,0.02,0.353,0.293,0.111,0.0505
IL-15,cDC1,52,0,0.0241,0.327,0.346,0.135,0.0962
IL-15,cDC2,82,0,0.00403,0.0469,0.122,0,0
IL-15,pDC,13,0,0.0272,0.138,0.308,0.231,0.154
IL-15,Fibroblast,98,0.00472,0.0359,0.529,0.367,0.184,0.112
IL-15,Keratinocyte,93,0.00441,0.0227,0.387,0.344,0.14,0.0645
IL-15,Vascular,101,0.00392,0.0177,0.386,0.347,0.099,0.0297
IL-18,CD4_malignant,78,0.000862,0.0139,0.248,0.231,0.0769,0.0385
IL-18,CD4_reactive,82,0.00528,0.0136,0.207,0.366,0.0488,0.0244
IL-18,CD4_HC,17,0,0.00313,0.0317,0.0588,0,0
IL-18,CD8,84,0,0.00965,0.13,0.238,0.0595,0.0238
IL-18,CD4_Treg,76,0,0.00824,0.328,0.158,0.0263,0.0132
IL-18,NK,24,0.0431,0.0488,0.212,0.75,0.417,0.167
IL-18,Mono_Mac,99,0,0.00193,0.0753,0.0303,0.0101,0
IL-18,cDC1,52,0,0.00505,0.115,0.115,0.0192,0.0192
IL-18,cDC2,82,0,0.00394,0.05,0.159,0.0122,0
IL-18,pDC,13,0,0.0153,0.0476,0.462,0,0
IL-18,Fibroblast,98,0,0.003,0.0704,0.0816,0.0204,0
IL-18,Keratinocyte,93,0,0.00192,0.0482,0.0538,0,0
IL-18,Vascular,101,0,0.000862,0.0625,0.0099,0.0099,0
XCL1,CD4_malignant,78,0,0.00254,0.14,0.0128,0.0128,0.0128
XCL1,CD4_reactive,82,0,0.00136,0.0488,0.0366,0,0
XCL1,CD4_HC,17,0,0,0,0,0,0
XCL1,CD8,84,0,0.000836,0.0312,0.0238,0,0
XCL1,CD4_Treg,76,0,0.00337,0.202,0.0395,0.0132,0.0132
XCL1,NK,24,0,0.000197,0.00473,0,0,0
XCL1,Mono_Mac,99,0,0.00259,0.0495,0.0808,0,0
XCL1,cDC1,52,0.383,0.407,0.903,1,0.962,0.904
XCL1,cDC2,82,0,0.000791,0.0153,0.0244,0,0
XCL1,pDC,13,0,0.0129,0.0968,0.308,0.0769,0
XCL1,Fibroblast,98,0,0.00506,0.13,0.0918,0.0306,0.0102
XCL1,Keratinocyte,93,0.000247,0.00322,0.0789,0.0753,0.0108,0
XCL1,Vascular,101,0,0.00191,0.05,0.0495,0.0099,0
```

Detection of a two-chain complex is the product of two sparse detections, so it is the quantity most sensitive to sequencing depth in this report. Read it beside the per-arm depth in §1b: an arm with half the UMI will show less than half the reach for the same biology.

## §4b Gate 2 -- per donor

**What:** Every (arm, donor) group's reachable fraction, for each payload.

**How:** Same quantity as §4a before aggregation. `n_cells` is the group size; groups below 20 cells are not built at all.

**Table on disk:** `tables/pay51_gate2_reach.csv`

```csv
arm,donor,n_cells,IL-12,IL-18,IL-15,XCL1,CXCL9/10
CD4_HC,D13__SC327_HTO1,28,0,0,0,0,0.107
CD4_HC,D13__SC327_HTO2,179,0.00559,0,0.106,0,0.0838
CD4_HC,D13__SC374_HTO10,782,0.00128,0.00384,0.0806,0,0.0422
CD4_HC,D13__SC374_HTO9,212,0,0,0.0755,0,0.0755
CD4_HC,D14__SC405,47,0,0,0.17,0,0.0638
CD4_HC,D15__NS13,299,0.00334,0,0.0602,0,0.0535
CD4_HC,D15__NS14,442,0.00226,0.0317,0.294,0,0.086
CD4_HC,D15__NS16,88,0,0,0.0227,0,0.0341
CD4_HC,D15__NS17,293,0,0.00683,0.0171,0,0.0273
CD4_HC,D1__P112,241,0,0.00415,0.17,0,0.0871
CD4_HC,D1__P115,459,0,0.00218,0.113,0,0.133
CD4_HC,D1__P116,443,0,0.00451,0.151,0,0.142
CD4_HC,D1__P121,64,0,0,0.0469,0,0.0469
CD4_HC,D6__NOR1,51,0,0,0.0392,0,0.0784
CD4_HC,D6__NOR2,49,0,0,0.163,0,0.0408
CD4_HC,D6__NOR3,80,0,0,0.113,0,0.05
CD4_HC,D6__NOR4,61,0,0,0.082,0,0.131
CD4_Treg,B5__PT09,890,0.0618,0,0.442,0,0.264
CD4_Treg,B5__PT22,795,0.0189,0,0.342,0,0.277
CD4_Treg,B5__PT23,278,0,0,0.245,0,0.00719
CD4_Treg,D10__JONAK1,375,0,0,0.0693,0,0.0773
CD4_Treg,D11__P1,100,0.06,0,0.32,0,0.16
CD4_Treg,D11__P2,797,0.0514,0.00125,0.487,0,0.181
CD4_Treg,D11__P3,745,0.0148,0,0.419,0,0.0779
CD4_Treg,D11__P4,142,0.0352,0,0.289,0,0.0845
CD4_Treg,D11__P5,769,0.0078,0.0039,0.41,0,0.013
CD4_Treg,D11__P6,866,0.00808,0,0.635,0,0.0277
CD4_Treg,D13__MF17,129,0.0155,0,0.279,0,0.116
CD4_Treg,D13__MF18,207,0.058,0.0145,0.444,0,0.507
CD4_Treg,D13__MF21,26,0,0,0.308,0,0.462
CD4_Treg,D13__MF25,26,0.115,0,0.346,0,0.269
CD4_Treg,D15__MF24,269,0.0743,0,0.227,0,0.23
CD4_Treg,D15__MF26,103,0.0485,0,0.408,0,0.117
CD4_Treg,D15__MF28,202,0.0198,0.0198,0.47,0,0.0693
CD4_Treg,D15__MF30,125,0.112,0.032,0.784,0.008,0.384
CD4_Treg,D15__MF34,41,0.0244,0.0244,0.317,0,0.122
CD4_Treg,D15__MF35,405,0.0691,0.328,0.911,0.202,0.627
CD4_Treg,D15__MF36,830,0.155,0,0.377,0.0012,0.17
CD4_Treg,D15__MF40,220,0,0.00909,0.3,0,0.114
CD4_Treg,D15__MF48,143,0.042,0.00699,0.378,0,0.357
CD4_Treg,D15__NS14,31,0,0.0323,0.419,0,0.0968
CD4_Treg,D1__P115,27,0,0,0.148,0,0.259
CD4_Treg,D1__P171,353,0,0,0.21,0,0.0085
CD4_Treg,D1__P204,1357,0.00221,0.000737,0.21,0.000737,0.197
CD4_Treg,D1__P303,128,0.0156,0.0547,0.227,0,0.656
CD4_Treg,D1__P312,570,0.00175,0,0.114,0,0.195
CD4_Treg,D1__PGS,290,0.0069,0.00345,0.231,0,0.0552
CD4_Treg,D3__P105,47,0,0,0.128,0,0.128
CD4_Treg,D3__P192,340,0.0147,0,0.359,0,0.294
CD4_Treg,D3__P196,127,0,0,0.197,0,0.591
CD4_Treg,D3__P94,174,0.00575,0,0.178,0,0.299
CD4_Treg,D6__MF2,63,0,0,0.27,0,0.143
CD4_Treg,D6__SC157,45,0,0,0.378,0,0.222
CD4_Treg,D6__SC158,27,0,0,0.222,0,0.185
CD4_Treg,D7__MF309,965,0.00518,0,0.0622,0,0.19
CD4_Treg,D7__MF311,1052,0.000951,0,0.0323,0,0.0361
CD4_Treg,D7__P73,63,0,0,0.222,0,0.27
CD4_Treg,D7__P84,629,0.00795,0.00159,0.188,0,0.275
CD4_Treg,D8__P138,215,0.00465,0,0.126,0,0.0791
CD4_Treg,D8__P150,289,0,0.00346,0.083,0,0.111
[839 further rows omitted from this block]
```

## §5a Gate 3 -- endogenous ligand, by arm

**What:** Whether each payload's ligand is already produced in the lesion, and by which population.

**How:** Pseudobulk per (arm, donor): `mean_all` is the mean over that group's cells of log1p CP10K, `pct_expr` the fraction of cells with >= 1 count; the table reports the median and mean of those donor values and the fraction of donors above the 10% detection floor. Descriptive, no test. Myeloid, fibroblast, keratinocyte and vascular arms are included because CXCL9/10/11 are largely myeloid and CCL21 is lymphatic-endothelial -- without them an absent ligand cannot be told apart from an absent cell type. (Those two context arms are an addition to the brief.)

**Table on disk:** `tables/pay51_gate3_summary.csv`

```csv
gene,arm,n_donors,median_pct_expr,mean_pct_expr,median_mean_all,frac_donors_pos
CCL19,CD4_malignant,78,0.00627,0.032,0.00842,0.128
CCL19,CD4_reactive,82,0.00629,0.0285,0.0107,0.122
CCL19,CD4_HC,17,0.00341,0.00517,0.00592,0
CCL19,CD8,84,0.00372,0.0213,0.00655,0.0595
CCL19,CD4_Treg,76,0.00274,0.0198,0.00457,0.0395
CCL19,NK,24,0,0.0151,0,0.0833
CCL19,Mono_Mac,99,0.036,0.0684,0.0519,0.222
CCL19,cDC1,52,0.0297,0.048,0.04,0.154
CCL19,cDC2,82,0.0318,0.0458,0.0357,0.122
CCL19,pDC,13,0,0.0648,0,0.231
CCL19,Fibroblast,98,0.281,0.332,0.731,0.806
CCL19,Keratinocyte,93,0.0312,0.0644,0.0565,0.226
CCL19,Vascular,101,0.0297,0.057,0.0468,0.139
CCL21,CD4_malignant,78,0.00348,0.0124,0.00412,0.0128
CCL21,CD4_reactive,82,0.0028,0.0112,0.00596,0.0122
CCL21,CD4_HC,17,0,0.0119,0,0
CCL21,CD8,84,0,0.0101,0,0.0238
CCL21,CD4_Treg,76,0,0.00967,0,0
CCL21,NK,24,0,0.0048,0,0
CCL21,Mono_Mac,99,0.00476,0.0144,0.00792,0.0202
CCL21,cDC1,52,0,0.0169,0,0.0385
CCL21,cDC2,82,0,0.0148,0,0.0244
CCL21,pDC,13,0,0.0107,0,0
CCL21,Fibroblast,98,0.0312,0.0581,0.0366,0.163
CCL21,Keratinocyte,93,0.0119,0.0319,0.0188,0.0968
CCL21,Vascular,101,0.128,0.177,0.484,0.624
CD70,CD4_malignant,78,0.21,0.234,0.292,0.808
CD70,CD4_reactive,82,0.0498,0.068,0.0886,0.195
CD70,CD4_HC,17,0.0273,0.0295,0.0445,0
CD70,CD8,84,0.0737,0.0886,0.135,0.345
CD70,CD4_Treg,76,0.122,0.159,0.219,0.671
CD70,NK,24,0.0178,0.0341,0.0381,0.0833
CD70,Mono_Mac,99,0.00746,0.0164,0.00659,0.0101
CD70,cDC1,52,0.0111,0.0256,0.0125,0.0577
CD70,cDC2,82,0.0139,0.023,0.0141,0.0244
CD70,pDC,13,0.0323,0.0316,0.0174,0.0769
CD70,Fibroblast,98,0.0525,0.0744,0.0548,0.235
CD70,Keratinocyte,93,0.00877,0.0173,0.00865,0
CD70,Vascular,101,0.00588,0.0136,0.00582,0.0099
CXCL10,CD4_malignant,78,0.0125,0.0353,0.0163,0.103
CXCL10,CD4_reactive,82,0.00353,0.0118,0.00585,0.0122
CXCL10,CD4_HC,17,0,0.00198,0,0
CXCL10,CD8,84,9.46e-05,0.00975,0.000145,0.0238
CXCL10,CD4_Treg,76,0.0059,0.0241,0.00928,0.0789
CXCL10,NK,24,0,0.00901,0,0
CXCL10,Mono_Mac,99,0.0816,0.124,0.184,0.424
CXCL10,cDC1,52,0.0285,0.048,0.0349,0.154
CXCL10,cDC2,82,0.0558,0.105,0.124,0.329
CXCL10,pDC,13,0,0.0332,0,0.154
CXCL10,Fibroblast,98,0.0357,0.076,0.066,0.296
CXCL10,Keratinocyte,93,0.012,0.0498,0.0216,0.14
CXCL10,Vascular,101,0.0312,0.0619,0.0546,0.188
CXCL11,CD4_malignant,78,0.000272,0.0038,0.000313,0
CXCL11,CD4_reactive,82,0,0.00205,0,0
CXCL11,CD4_HC,17,0,0,0,0
CXCL11,CD8,84,0,0.00187,0,0
CXCL11,CD4_Treg,76,0,0.00225,0,0
CXCL11,NK,24,0,0.000131,0,0
CXCL11,Mono_Mac,99,0.0137,0.0351,0.0192,0.0707
CXCL11,cDC1,52,0,0.0068,0,0
CXCL11,cDC2,82,0.00757,0.0256,0.00815,0.0732
CXCL11,pDC,13,0,0.00427,0,0
CXCL11,Fibroblast,98,0.00891,0.0401,0.0137,0.122
CXCL11,Keratinocyte,93,0.00408,0.0279,0.00376,0.0753
CXCL11,Vascular,101,0.0314,0.0771,0.0447,0.267
CXCL9,CD4_malignant,78,0.00325,0.0158,0.00517,0.0385
CXCL9,CD4_reactive,82,0.00131,0.0158,0.00225,0.0366
CXCL9,CD4_HC,17,0,0.000705,0,0
CXCL9,CD8,84,0,0.0101,0,0.0119
CXCL9,CD4_Treg,76,0,0.00985,0,0.0132
CXCL9,NK,24,0,0.00821,0,0
CXCL9,Mono_Mac,99,0.0824,0.16,0.2,0.465
CXCL9,cDC1,52,0.127,0.152,0.219,0.577
CXCL9,cDC2,82,0.0597,0.125,0.138,0.402
CXCL9,pDC,13,0.027,0.0591,0.0377,0.154
CXCL9,Fibroblast,98,0.0633,0.16,0.123,0.388
CXCL9,Keratinocyte,93,0.00828,0.0397,0.0128,0.086
CXCL9,Vascular,101,0.0227,0.0608,0.0414,0.188
FLT3LG,CD4_malignant,78,0.185,0.248,0.248,0.859
FLT3LG,CD4_reactive,82,0.284,0.29,0.486,0.963
FLT3LG,CD4_HC,17,0.317,0.299,0.541,0.941
FLT3LG,CD8,84,0.146,0.167,0.239,0.774
FLT3LG,CD4_Treg,76,0.24,0.248,0.346,0.895
FLT3LG,NK,24,0.113,0.121,0.194,0.583
FLT3LG,Mono_Mac,99,0.0413,0.0529,0.0462,0.131
FLT3LG,cDC1,52,0.0244,0.0305,0.0202,0.0385
FLT3LG,cDC2,82,0.0468,0.0553,0.0422,0.146
FLT3LG,pDC,13,0.0465,0.0562,0.0312,0.231
FLT3LG,Fibroblast,98,0.279,0.305,0.343,0.939
FLT3LG,Keratinocyte,93,0.081,0.0872,0.0643,0.344
FLT3LG,Vascular,101,0.115,0.126,0.115,0.663
IL12A,CD4_malignant,78,0.00104,0.00622,0.00111,0.0128
IL12A,CD4_reactive,82,0.00119,0.00216,0.00132,0
IL12A,CD4_HC,17,0,0.000711,0,0
IL12A,CD8,84,0.00231,0.00454,0.00386,0
IL12A,CD4_Treg,76,0,0.00281,0,0
IL12A,NK,24,0,0.00163,0,0
IL12A,Mono_Mac,99,0,0.00114,0,0
IL12A,cDC1,52,0,0.000499,0,0
IL12A,cDC2,82,0,0.00139,0,0
IL12A,pDC,13,0,0.0016,0,0
IL12A,Fibroblast,98,0.000835,0.00265,0.00062,0
IL12A,Keratinocyte,93,0,0.00196,0,0
IL12A,Vascular,101,0.00693,0.0103,0.00598,0
IL12B,CD4_malignant,78,0,0.000231,0,0
IL12B,CD4_reactive,82,0,0.000211,0,0
IL12B,CD4_HC,17,0,0,0,0
IL12B,CD8,84,0,2.03e-05,0,0
IL12B,CD4_Treg,76,0,0.000321,0,0
IL12B,NK,24,0,0,0,0
IL12B,Mono_Mac,99,0,0.000384,0,0
IL12B,cDC1,52,0.0189,0.0308,0.0258,0.0577
IL12B,cDC2,82,0,0.00035,0,0
IL12B,pDC,13,0,0.0014,0,0
IL12B,Fibroblast,98,0,0.00046,0,0
IL12B,Keratinocyte,93,0,0.000244,0,0
IL12B,Vascular,101,0,0.00032,0,0
IL15,CD4_malignant,78,0.00661,0.0199,0.00685,0.0256
IL15,CD4_reactive,82,0.00776,0.0143,0.0148,0.0122
IL15,CD4_HC,17,0,0.00546,0,0
IL15,CD8,84,0.0131,0.02,0.0232,0.0238
IL15,CD4_Treg,76,0.00477,0.0148,0.00623,0.0132
IL15,NK,24,0.0179,0.0194,0.0305,0
IL15,Mono_Mac,99,0.0916,0.1,0.102,0.434
IL15,cDC1,52,0.143,0.172,0.162,0.673
IL15,cDC2,82,0.139,0.175,0.14,0.732
IL15,pDC,13,0.143,0.132,0.119,0.692
IL15,Fibroblast,98,0.105,0.132,0.111,0.541
IL15,Keratinocyte,93,0.0247,0.0399,0.0204,0.0753
IL15,Vascular,101,0.155,0.179,0.167,0.802
IL18,CD4_malignant,78,0.00192,0.00605,0.00289,0
IL18,CD4_reactive,82,0.00764,0.0135,0.011,0
IL18,CD4_HC,17,0.00669,0.0065,0.0112,0
IL18,CD8,84,0.000822,0.00552,0.000973,0
IL18,CD4_Treg,76,0,0.00576,0,0.0132
IL18,NK,24,0.0342,0.0625,0.0623,0.208
IL18,Mono_Mac,99,0.181,0.229,0.254,0.838
IL18,cDC1,52,0.168,0.179,0.185,0.731
IL18,cDC2,82,0.345,0.361,0.405,0.963
IL18,pDC,13,0.333,0.386,0.446,1
IL18,Fibroblast,98,0.0258,0.0341,0.0281,0.0612
IL18,Keratinocyte,93,0.356,0.344,0.426,0.925
IL18,Vascular,101,0.00645,0.0107,0.00564,0.0099
IL18BP,CD4_malignant,78,0.0229,0.0494,0.0262,0.192
IL18BP,CD4_reactive,82,0.0272,0.0417,0.0391,0.0854
IL18BP,CD4_HC,17,0.0137,0.0151,0.0171,0
IL18BP,CD8,84,0.0295,0.0405,0.0485,0.0833
IL18BP,CD4_Treg,76,0.0322,0.0424,0.0464,0.105
IL18BP,NK,24,0.034,0.0363,0.0449,0.0417
IL18BP,Mono_Mac,99,0.222,0.244,0.279,0.768
IL18BP,cDC1,52,0.0674,0.0905,0.0663,0.308
IL18BP,cDC2,82,0.255,0.265,0.28,0.878
IL18BP,pDC,13,0.1,0.125,0.0692,0.538
IL18BP,Fibroblast,98,0.0799,0.134,0.0744,0.459
IL18BP,Keratinocyte,93,0.0157,0.0352,0.012,0.129
IL18BP,Vascular,101,0.2,0.216,0.222,0.772
TNFSF4,CD4_malignant,78,0.0416,0.0722,0.0459,0.231
TNFSF4,CD4_reactive,82,0.0153,0.0218,0.0231,0
TNFSF4,CD4_HC,17,0.0164,0.02,0.0266,0
TNFSF4,CD8,84,0.0137,0.0207,0.021,0.0357
TNFSF4,CD4_Treg,76,0.00989,0.0192,0.012,0.0132
TNFSF4,NK,24,0.023,0.0332,0.0481,0.0833
TNFSF4,Mono_Mac,99,0,0.00353,0,0
TNFSF4,cDC1,52,0,0.00466,0,0
TNFSF4,cDC2,82,0,0.00645,0,0
TNFSF4,pDC,13,0.133,0.109,0.148,0.538
TNFSF4,Fibroblast,98,0.0151,0.0255,0.0133,0.0306
TNFSF4,Keratinocyte,93,0.00185,0.00632,0.00118,0.0108
TNFSF4,Vascular,101,0.000895,0.00448,0.000951,0
TNFSF9,CD4_malignant,78,0.084,0.157,0.116,0.449
TNFSF9,CD4_reactive,82,0.0591,0.0823,0.0976,0.354
TNFSF9,CD4_HC,17,0.061,0.0668,0.0978,0.294
TNFSF9,CD8,84,0.108,0.182,0.197,0.536
TNFSF9,CD4_Treg,76,0.0219,0.0559,0.0279,0.145
TNFSF9,NK,24,0.0798,0.155,0.17,0.458
TNFSF9,Mono_Mac,99,0.0337,0.058,0.0396,0.202
TNFSF9,cDC1,52,0.0295,0.0877,0.0326,0.308
TNFSF9,cDC2,82,0.0615,0.0859,0.0658,0.354
TNFSF9,pDC,13,0.179,0.199,0.168,0.615
TNFSF9,Fibroblast,98,0.047,0.0813,0.0464,0.286
TNFSF9,Keratinocyte,93,0.0421,0.0744,0.0331,0.28
TNFSF9,Vascular,101,0.0397,0.0688,0.0363,0.257
XCL1,CD4_malignant,78,0.0109,0.109,0.0202,0.231
XCL1,CD4_reactive,82,0.0927,0.113,0.277,0.451
XCL1,CD4_HC,17,0.138,0.134,0.442,0.706
XCL1,CD8,84,0.102,0.126,0.262,0.524
XCL1,CD4_Treg,76,0.000256,0.00883,0.000325,0
XCL1,NK,24,0.624,0.522,1.78,0.958
XCL1,Mono_Mac,99,0,0.00623,0,0.0101
XCL1,cDC1,52,0,0.00726,0,0
XCL1,cDC2,82,0,0.00445,0,0.0122
XCL1,pDC,13,0,0.0114,0,0
XCL1,Fibroblast,98,0.00349,0.0135,0.0054,0.0204
XCL1,Keratinocyte,93,0.00295,0.01,0.00448,0.0215
XCL1,Vascular,101,0.00165,0.00691,0.00234,0
XCL2,CD4_malignant,78,0.00572,0.0848,0.0105,0.154
XCL2,CD4_reactive,82,0.0879,0.111,0.216,0.476
XCL2,CD4_HC,17,0.082,0.101,0.278,0.412
XCL2,CD8,84,0.146,0.151,0.339,0.679
XCL2,CD4_Treg,76,0,0.00439,0,0
[8 further rows omitted from this block]
```

## §5b Gate 3 -- IL18BP, the decoy

**What:** IL-18 beside its decoy receptor IL18BP, which is what decides between native IL-18 and a decoy-resistant variant as the payload.

**How:** Same quantity as §5a, restricted to the two genes so the comparison is not buried. IL18BP present in the lesion means a native IL-18 payload is neutralised before it acts; the arm carrying it says which cell does the neutralising.

**Table on disk:** `tables/pay51_gate3_summary.csv`

```csv
gene,arm,n_donors,median_pct_expr,mean_pct_expr,median_mean_all,frac_donors_pos
IL18,CD4_malignant,78,0.00192,0.00605,0.00289,0
IL18,CD4_reactive,82,0.00764,0.0135,0.011,0
IL18,CD4_HC,17,0.00669,0.0065,0.0112,0
IL18,CD8,84,0.000822,0.00552,0.000973,0
IL18,CD4_Treg,76,0,0.00576,0,0.0132
IL18,NK,24,0.0342,0.0625,0.0623,0.208
IL18,Mono_Mac,99,0.181,0.229,0.254,0.838
IL18,cDC1,52,0.168,0.179,0.185,0.731
IL18,cDC2,82,0.345,0.361,0.405,0.963
IL18,pDC,13,0.333,0.386,0.446,1
IL18,Fibroblast,98,0.0258,0.0341,0.0281,0.0612
IL18,Keratinocyte,93,0.356,0.344,0.426,0.925
IL18,Vascular,101,0.00645,0.0107,0.00564,0.0099
IL18BP,CD4_malignant,78,0.0229,0.0494,0.0262,0.192
IL18BP,CD4_reactive,82,0.0272,0.0417,0.0391,0.0854
IL18BP,CD4_HC,17,0.0137,0.0151,0.0171,0
IL18BP,CD8,84,0.0295,0.0405,0.0485,0.0833
IL18BP,CD4_Treg,76,0.0322,0.0424,0.0464,0.105
IL18BP,NK,24,0.034,0.0363,0.0449,0.0417
IL18BP,Mono_Mac,99,0.222,0.244,0.279,0.768
IL18BP,cDC1,52,0.0674,0.0905,0.0663,0.308
IL18BP,cDC2,82,0.255,0.265,0.28,0.878
IL18BP,pDC,13,0.1,0.125,0.0692,0.538
IL18BP,Fibroblast,98,0.0799,0.134,0.0744,0.459
IL18BP,Keratinocyte,93,0.0157,0.0352,0.012,0.129
IL18BP,Vascular,101,0.2,0.216,0.222,0.772
```

## §6a Sensitivity -- excluding the cultured donors

**What:** Gate 1 recomputed with the ex-vivo cultured donors removed, against the headline.

**How:** Donors flagged `is_cultured` by `target_cohort` (built from `clin_ex_vivo`): D15__MF22, D15__MF28, D15__MF30, D15__MF31, D15__MF34, D15__MF35, D15__MF40; of those, D15__MF28, D15__MF30, D15__MF34, D15__MF35 carry both CD4 arms and therefore affect the paired test. Columns prefixed `head_` are the headline values of §3; `d_` is this analysis minus the headline; `sign_flip_q05` flags a gene that crosses q = 0.05 in either direction.

**Table on disk:** `tables/pay51_sens_no_cultured.csv`

```csv
gene,family,log2fc_paired,q,log2fc_detect,q_detect,n_donors_paired,head_log2fc_paired,head_q,head_log2fc_detect,head_q_detect,head_n_donors_paired,d_log2fc_paired,d_log2fc_detect,sign_flip_q05
CD27,A_permitted,-0.0183,0.599,0.222,0.158,71,-0.359,0.78,-0.013,0.273,75,0.341,0.235,False
CD40,A_permitted,-0.11,0.000948,-0.0465,0.028,71,-0.125,0.000276,-0.0465,0.0147,75,0.0154,0,False
IFNGR1,A_permitted,-0.301,0.00506,0.00329,0.354,71,-0.21,0.00585,0.0483,0.178,75,-0.0912,-0.045,False
IFNGR2,A_permitted,0.315,0.00252,0.479,2.83e-05,71,0.294,0.00703,0.46,0.000103,75,0.0211,0.0191,False
IL10RA,A_permitted,-0.815,1.06e-08,-0.559,6.29e-05,71,-0.783,3.33e-09,-0.472,9.75e-05,75,-0.0316,-0.0868,False
IL12RB1,A_permitted,-0.238,0.138,0.11,0.0774,71,-0.152,0.191,0.146,0.0312,75,-0.0858,-0.0357,False
IL12RB2,A_permitted,-0.41,0.00014,-0.212,0.00814,71,-0.449,3.42e-05,-0.212,0.00657,75,0.0392,0,False
IL15RA,A_permitted,-0.0328,0.187,0.0401,0.158,71,-0.0328,0.263,0.0621,0.0434,75,0,-0.022,False
IL18R1,A_permitted,-0.331,0.000875,-0.0793,0.354,71,-0.325,0.0019,-0.0793,0.473,75,-0.00653,0,False
IL18RAP,A_permitted,-0.591,0.000428,-0.26,0.0258,71,-0.564,0.00105,-0.257,0.0728,75,-0.0267,-0.00342,False
IL23R,A_permitted,-0.0862,0.987,0,0.378,71,-0.0862,0.822,0,0.503,75,0,0,False
IL2RB,A_permitted,0.141,0.312,0.365,0.00044,71,0.141,0.263,0.327,0.000249,75,0,0.0375,False
IL2RG,A_permitted,0.19,0.00301,0.177,5.13e-06,71,0.176,0.004,0.17,1.24e-05,75,0.0135,0.00737,False
IL7R,A_permitted,-0.758,9.11e-10,-0.372,4.95e-05,71,-0.775,1.52e-10,-0.392,1.38e-05,75,0.0168,0.0202,False
TNFRSF1A,A_permitted,0.0164,0.709,0.164,0.00242,71,0.0136,0.815,0.164,0.00194,75,0.00277,0,False
TNFRSF1B,A_permitted,-0.15,0.138,0.109,0.231,71,-0.15,0.191,0.109,0.18,75,0,0,False
TNFRSF9,A_permitted,-0.273,0.057,-0.132,0.757,71,-0.266,0.102,-0.111,0.996,75,-0.0076,-0.0212,False
IL10RB,B_disqualified,0.372,2.06e-06,0.654,2.97e-10,71,0.371,6.65e-06,0.645,4.78e-10,75,0.00155,0.0082,False
IL13RA1,B_disqualified,0.0132,0.201,0.016,0.00573,71,0.0132,0.129,0.0306,0.00139,75,0,-0.0147,False
IL13RA2,B_disqualified,0,0.00989,0,0.000284,71,0,0.0459,0,0.00339,75,0,0,False
IL21R,B_disqualified,0.173,0.0108,0.528,1.11e-05,71,0.203,0.00295,0.534,1.55e-06,75,-0.03,-0.00618,False
IL27RA,B_disqualified,0.326,0.000117,0.525,5.89e-09,71,0.313,0.000321,0.479,9.81e-09,75,0.0125,0.0464,False
IL2RA,B_disqualified,-0.237,0.132,-0.00686,0.269,71,-0.172,0.329,0.0171,0.141,75,-0.0654,-0.024,False
IL4R,B_disqualified,0.287,0.00103,0.565,1.16e-07,71,0.286,0.000638,0.563,5.89e-08,75,0.000831,0.002,False
IL6R,B_disqualified,0.614,5.24e-05,0.764,3.32e-08,71,0.614,1.78e-05,0.778,9.81e-09,75,0,-0.0137,False
IL9R,B_disqualified,0.472,0.00181,0.541,5.05e-05,71,0.472,0.000638,0.541,1.43e-05,75,0,0,False
TNFRSF18,B_disqualified,0.343,0.0108,0.442,5.05e-05,71,0.343,0.00595,0.442,2.08e-05,75,0,0,False
TNFRSF25,B_disqualified,0.747,6.08e-05,0.888,1.16e-07,71,0.698,5.87e-05,0.821,7.66e-08,75,0.0496,0.0674,False
TNFRSF4,B_disqualified,0.542,0.00165,0.538,1.24e-05,71,0.542,0.0013,0.538,7.06e-06,75,0,0,False
TNFRSF8,B_disqualified,0.302,0.000465,0.43,1.19e-06,71,0.332,0.000116,0.455,1.7e-07,75,-0.0299,-0.0246,False
CCL19,C_axis,-0.032,0.0101,0,0.727,71,-0.0028,0.0225,0,0.853,75,-0.0292,0,False
CCL21,C_axis,-0.0126,0.0217,0,0.83,71,-0.00858,0.0225,0,0.853,75,-0.00403,0,False
CCR4,C_axis,0.41,0.045,0.591,0.00101,71,0.324,0.0645,0.568,0.00138,75,0.0864,0.0222,True
CCR5,C_axis,-0.428,0.000837,-0.279,0.00604,71,-0.409,0.000257,-0.279,0.00232,75,-0.0185,0,False
CCR7,C_axis,0.0611,0.122,0.412,0.00604,71,-0.0856,0.269,0.197,0.018,75,0.147,0.215,False
CD70,C_axis,1.1,1.48e-10,1.21,9.31e-12,71,1.1,2.72e-11,1.22,1.97e-12,75,0,-0.0102,False
CX3CR1,C_axis,-0.246,0.000742,-0.112,0.00829,71,-0.246,0.00021,-0.112,0.00327,75,0,0,False
CXCL10,C_axis,0.0999,0.000837,0.137,1.86e-05,71,0.0999,0.000721,0.137,7.51e-06,75,0,0,False
CXCL11,C_axis,0,0.241,0,0.0233,71,0,0.342,0,0.0453,75,0,0,False
CXCL9,C_axis,0,0.0989,0,0.716,71,0,0.0728,0,0.853,75,0,0,False
CXCR3,C_axis,-1.28,0.071,-0.926,0.307,71,-1.24,0.0645,-0.819,0.482,75,-0.04,-0.107,False
FLT3LG,C_axis,-0.585,1.28e-05,-0.21,0.0689,71,-0.618,2.38e-06,-0.21,0.0362,75,0.0334,0,False
IL12A,C_axis,0,0.613,0,0.06,71,0,0.897,0,0.096,75,0,0,False
IL12B,C_axis,0,0.473,0,0.275,71,0,0.331,0,0.165,75,0,0,False
IL15,C_axis,-0.0325,0.574,0,0.405,71,-0.0325,0.743,0,0.35,75,0,0,False
IL18,C_axis,-0.232,2.24e-08,-0.12,1.92e-07,71,-0.215,2.03e-08,-0.112,1.58e-07,75,-0.0173,-0.00746,False
IL18BP,C_axis,-0.224,0.0681,-0.0609,0.872,71,-0.224,0.0669,-0.0609,0.853,75,0,0,False
S1PR1,C_axis,-0.905,0.000837,-0.457,0.0585,71,-0.934,0.00021,-0.596,0.0158,75,0.0281,0.14,False
TNFSF4,C_axis,0.289,0.000837,0.295,2.47e-05,71,0.289,0.000333,0.295,8.43e-06,75,0,0,False
TNFSF9,C_axis,0.266,0.007,0.461,2.16e-05,71,0.32,0.00377,0.485,1.17e-05,75,-0.0542,-0.0241,False
XCL1,C_axis,-1.61,0.00389,-0.842,0.0161,71,-1.56,0.00152,-0.819,0.00805,75,-0.045,-0.0228,False
XCL2,C_axis,-1.79,0.000499,-1.03,0.00101,71,-1.74,0.000174,-0.953,0.000407,75,-0.0575,-0.0784,False
XCR1,C_axis,0,0.574,0,0.307,71,0,0.342,0,0.156,75,0,0,False
CD2,audit,-0.146,0.0424,0.0676,0.909,71,-0.146,0.0266,0.0676,0.975,75,0,0,False
CD3E,audit,-0.0939,0.0443,0.0873,0.0223,71,-0.0939,0.0266,0.0782,0.0487,75,0,0.00909,False
CD7,audit,-3.12,1.26e-09,-2.37,2.39e-10,71,-2.85,3.02e-10,-2.28,6e-11,75,-0.268,-0.0945,False
GAPDH,audit,0.149,0.000679,0.0499,0.0223,41,0.16,0.000121,0.0384,0.0432,45,-0.0114,0.0115,False
PTPRC,audit,-0.0292,0.205,0.0557,0.0223,71,-0.0324,0.0559,0.0515,0.0826,75,0.00316,0.0042,False
```

## §6b Sensitivity -- donor-arms below 60 cells removed

**What:** Gate 1 recomputed with every donor-arm under 60 cells dropped, against the headline.

**How:** The headline floor is 20 cells per donor-arm; here it is 60. A small arm estimates `pct_expr` from few cells, so it is both noisy and biased toward 0 for a sparse gene. Same columns as §6a.

**Table on disk:** `tables/pay51_sens_min60cells.csv`

```csv
gene,family,log2fc_paired,q,log2fc_detect,q_detect,n_donors_paired,head_log2fc_paired,head_q,head_log2fc_detect,head_q_detect,head_n_donors_paired,d_log2fc_paired,d_log2fc_detect,sign_flip_q05
CD27,A_permitted,-0.434,0.905,-0.0294,0.444,70,-0.359,0.78,-0.013,0.273,75,-0.0751,-0.0165,False
CD40,A_permitted,-0.14,0.000472,-0.0508,0.0219,70,-0.125,0.000276,-0.0465,0.0147,75,-0.0151,-0.00427,False
IFNGR1,A_permitted,-0.159,0.023,0.0727,0.0773,70,-0.21,0.00585,0.0483,0.178,75,0.0508,0.0244,False
IFNGR2,A_permitted,0.287,0.0179,0.456,0.000427,70,0.294,0.00703,0.46,0.000103,75,-0.00707,-0.00444,False
IL10RA,A_permitted,-0.75,3.28e-08,-0.373,0.000514,70,-0.783,3.33e-09,-0.472,9.75e-05,75,0.033,0.0989,False
IL12RB1,A_permitted,-0.195,0.129,0.141,0.0551,70,-0.152,0.191,0.146,0.0312,75,-0.0429,-0.00465,False
IL12RB2,A_permitted,-0.43,0.000212,-0.193,0.00683,70,-0.449,3.42e-05,-0.212,0.00657,75,0.0196,0.0198,False
IL15RA,A_permitted,-0.0303,0.441,0.087,0.0215,70,-0.0328,0.263,0.0621,0.0434,75,0.00247,0.0249,False
IL18R1,A_permitted,-0.322,0.00521,-0.0757,0.588,70,-0.325,0.0019,-0.0793,0.473,75,0.00223,0.00361,False
IL18RAP,A_permitted,-0.55,0.00162,-0.253,0.0905,70,-0.564,0.00105,-0.257,0.0728,75,0.0142,0.0037,False
IL23R,A_permitted,-0.0977,0.848,-0.00683,0.536,70,-0.0862,0.822,0,0.503,75,-0.0115,-0.00683,False
IL2RB,A_permitted,0.15,0.182,0.379,0.000173,70,0.141,0.263,0.327,0.000249,75,0.00829,0.0515,False
IL2RG,A_permitted,0.198,0.000472,0.178,1.74e-06,70,0.176,0.004,0.17,1.24e-05,75,0.0213,0.00803,False
IL7R,A_permitted,-0.789,1.18e-09,-0.382,6.53e-05,70,-0.775,1.52e-10,-0.392,1.38e-05,75,-0.0142,0.0101,False
TNFRSF1A,A_permitted,-0.00678,0.848,0.199,0.00206,70,0.0136,0.815,0.164,0.00194,75,-0.0204,0.0352,False
TNFRSF1B,A_permitted,-0.0333,0.441,0.117,0.0571,70,-0.15,0.191,0.109,0.18,75,0.117,0.00808,False
TNFRSF9,A_permitted,-0.239,0.246,-0.0313,0.588,70,-0.266,0.102,-0.111,0.996,75,0.0265,0.0795,False
IL10RB,B_disqualified,0.333,1.62e-05,0.627,2.65e-09,70,0.371,6.65e-06,0.645,4.78e-10,75,-0.0382,-0.0189,False
IL13RA1,B_disqualified,0.0104,0.231,0.0254,0.00345,70,0.0132,0.129,0.0306,0.00139,75,-0.00285,-0.00521,False
IL13RA2,B_disqualified,0,0.0459,0,0.00345,70,0,0.0459,0,0.00339,75,0,0,False
IL21R,B_disqualified,0.222,0.00068,0.536,3.58e-07,70,0.203,0.00295,0.534,1.55e-06,75,0.0183,0.00252,False
IL27RA,B_disqualified,0.313,0.000436,0.465,4.72e-08,70,0.313,0.000321,0.479,9.81e-09,75,-0.000521,-0.0143,False
IL2RA,B_disqualified,-0.166,0.568,0.0886,0.0692,70,-0.172,0.329,0.0171,0.141,75,0.00624,0.0714,False
IL4R,B_disqualified,0.294,0.000436,0.564,9.88e-08,70,0.286,0.000638,0.563,5.89e-08,75,0.00854,0.001,False
IL6R,B_disqualified,0.61,4e-05,0.749,6.71e-08,70,0.614,1.78e-05,0.778,9.81e-09,75,-0.0048,-0.0284,False
IL9R,B_disqualified,0.526,0.000333,0.636,5.38e-06,70,0.472,0.000638,0.541,1.43e-05,75,0.0542,0.0959,False
TNFRSF18,B_disqualified,0.433,0.000614,0.48,6.63e-07,70,0.343,0.00595,0.442,2.08e-05,75,0.0908,0.0377,False
TNFRSF25,B_disqualified,0.686,0.000196,0.821,3.95e-07,70,0.698,5.87e-05,0.821,7.66e-08,75,-0.012,-6.84e-05,False
TNFRSF4,B_disqualified,0.576,0.00195,0.543,1.24e-05,70,0.542,0.0013,0.538,7.06e-06,75,0.0344,0.00509,False
TNFRSF8,B_disqualified,0.477,5.42e-05,0.456,1.26e-07,70,0.332,0.000116,0.455,1.7e-07,75,0.145,0.00105,False
CCL19,C_axis,-0.0136,0.0407,0.00251,0.893,70,-0.0028,0.0225,0,0.853,75,-0.0108,0.00251,False
CCL21,C_axis,-0.0162,0.0158,0,0.829,70,-0.00858,0.0225,0,0.853,75,-0.00766,0,False
CCR4,C_axis,0.257,0.127,0.468,0.00477,70,0.324,0.0645,0.568,0.00138,75,-0.067,-0.1,False
CCR5,C_axis,-0.418,0.000296,-0.289,0.0021,70,-0.409,0.000257,-0.279,0.00232,75,-0.00923,-0.0107,False
CCR7,C_axis,-0.142,0.584,0.138,0.0671,70,-0.0856,0.269,0.197,0.018,75,-0.0567,-0.0595,False
CD70,C_axis,1.12,2.17e-10,1.22,1.37e-11,70,1.1,2.72e-11,1.22,1.97e-12,75,0.0142,0.000287,False
CX3CR1,C_axis,-0.249,0.000208,-0.119,0.00364,70,-0.246,0.00021,-0.112,0.00327,75,-0.00383,-0.00686,False
CXCL10,C_axis,0.0682,0.00295,0.106,2.74e-05,70,0.0999,0.000721,0.137,7.51e-06,75,-0.0317,-0.0305,False
CXCL11,C_axis,0,0.385,0,0.0567,70,0,0.342,0,0.0453,75,0,0,False
CXCL9,C_axis,0,0.0275,0,0.758,70,0,0.0728,0,0.853,75,0,0,True
CXCR3,C_axis,-1.15,0.102,-0.755,0.758,70,-1.24,0.0645,-0.819,0.482,75,0.0884,0.0642,False
FLT3LG,C_axis,-0.667,6.68e-06,-0.212,0.0528,70,-0.618,2.38e-06,-0.21,0.0362,75,-0.0487,-0.00203,False
IL12A,C_axis,0,0.923,0,0.115,70,0,0.897,0,0.096,75,0,0,False
IL12B,C_axis,0,0.214,0,0.112,70,0,0.331,0,0.165,75,0,0,False
IL15,C_axis,-0.0333,0.672,0.00641,0.337,70,-0.0325,0.743,0,0.35,75,-0.000767,0.00641,False
IL18,C_axis,-0.233,4.2e-08,-0.117,3.09e-07,70,-0.215,2.03e-08,-0.112,1.58e-07,75,-0.0183,-0.00478,False
IL18BP,C_axis,-0.234,0.0848,-0.065,0.883,70,-0.224,0.0669,-0.0609,0.853,75,-0.0103,-0.00407,False
S1PR1,C_axis,-1.09,0.000113,-0.725,0.00672,70,-0.934,0.00021,-0.596,0.0158,75,-0.153,-0.128,False
TNFSF4,C_axis,0.311,0.000549,0.3,1.57e-05,70,0.289,0.000333,0.295,8.43e-06,75,0.0219,0.00536,False
TNFSF9,C_axis,0.364,0.000923,0.52,3.54e-06,70,0.32,0.00377,0.485,1.17e-05,75,0.0436,0.0354,False
XCL1,C_axis,-1.51,0.00611,-0.782,0.0191,70,-1.56,0.00152,-0.819,0.00805,75,0.0595,0.0369,False
XCL2,C_axis,-1.54,0.000549,-0.919,0.00101,70,-1.74,0.000174,-0.953,0.000407,75,0.197,0.0343,False
XCR1,C_axis,0,0.355,0,0.134,70,0,0.342,0,0.156,75,0,0,False
CD2,audit,-0.0771,0.064,0.153,0.75,70,-0.146,0.0266,0.0676,0.975,75,0.0689,0.0852,True
CD3E,audit,-0.0835,0.064,0.0841,0.0541,70,-0.0939,0.0266,0.0782,0.0487,75,0.0103,0.00597,True
CD7,audit,-2.85,1.61e-09,-2.25,2.96e-10,70,-2.85,3.02e-10,-2.28,6e-11,75,0.00221,0.0295,False
GAPDH,audit,0.192,1.98e-05,0.0522,0.0023,41,0.16,0.000121,0.0384,0.0432,45,0.0315,0.0138,False
PTPRC,audit,-0.0408,0.0446,0.0543,0.0759,70,-0.0324,0.0559,0.0515,0.0826,75,-0.00836,0.00282,True
```

## §6c Sensitivity -- stage, with study as a covariate

**What:** Whether the malignant-vs-reactive effect differs between early and advanced disease once the cohort of origin is held fixed.

**How:** Per gene, ordinary least squares of the donor-level log2FC on an early-vs-advanced indicator plus one dummy per study; `beta_early_vs_advanced` is the stage coefficient with study held fixed, `p` its two-sided t test, `q` BH over the genes of this table. `study_absorbed` is True where study and stage are collinear within the genes' donors and the study dummies had to be dropped.

**Table on disk:** `tables/pay51_sens_stage_study.csv`

```csv
gene,family,beta_early_vs_advanced,se,t,p,n_early,n_advanced,study_absorbed,q
IL12RB1,A_permitted,0.248,0.231,1.08,0.285,18,55,False,0.686
IL12RB2,A_permitted,0.245,0.337,0.727,0.47,18,55,False,0.758
IL18R1,A_permitted,0.337,0.291,1.16,0.251,18,55,False,0.686
IL18RAP,A_permitted,0.483,0.292,1.65,0.103,18,55,False,0.458
IL2RB,A_permitted,0.363,0.343,1.06,0.294,18,55,False,0.686
IL2RG,A_permitted,0.297,0.111,2.69,0.00927,18,55,False,0.134
IL15RA,A_permitted,0.035,0.209,0.168,0.867,18,55,False,0.932
IL7R,A_permitted,0.371,0.368,1.01,0.318,18,55,False,0.686
IL10RA,A_permitted,0.335,0.325,1.03,0.306,18,55,False,0.686
CD40,A_permitted,0.104,0.137,0.758,0.451,18,55,False,0.748
TNFRSF9,A_permitted,1.3,0.449,2.89,0.00536,18,55,False,0.115
CD27,A_permitted,-1.26,0.57,-2.21,0.0312,18,55,False,0.226
IFNGR1,A_permitted,0.225,0.229,0.979,0.331,18,55,False,0.686
IFNGR2,A_permitted,-0.962,0.338,-2.85,0.00595,18,55,False,0.115
IL23R,A_permitted,-0.194,0.301,-0.644,0.522,18,55,False,0.79
TNFRSF1A,A_permitted,0.203,0.228,0.891,0.376,18,55,False,0.701
TNFRSF1B,A_permitted,0.184,0.312,0.588,0.558,18,55,False,0.79
IL21R,B_disqualified,0.073,0.32,0.228,0.82,18,55,False,0.902
IL4R,B_disqualified,0.175,0.289,0.607,0.546,18,55,False,0.79
IL6R,B_disqualified,-0.318,0.336,-0.946,0.348,18,55,False,0.686
IL27RA,B_disqualified,-0.123,0.2,-0.616,0.54,18,55,False,0.79
IL9R,B_disqualified,-0.118,0.467,-0.254,0.801,18,55,False,0.902
IL10RB,B_disqualified,-0.00573,0.18,-0.0317,0.975,18,55,False,0.975
IL13RA1,B_disqualified,0.329,0.173,1.9,0.062,18,55,False,0.36
IL13RA2,B_disqualified,-0.155,0.0941,-1.65,0.104,18,55,False,0.458
TNFRSF25,B_disqualified,-0.133,0.424,-0.315,0.754,18,55,False,0.902
TNFRSF4,B_disqualified,-0.143,0.474,-0.3,0.765,18,55,False,0.902
TNFRSF8,B_disqualified,0.319,0.316,1.01,0.316,18,55,False,0.686
TNFRSF18,B_disqualified,0.7,0.396,1.76,0.0827,18,55,False,0.436
IL2RA,B_disqualified,-0.135,0.374,-0.361,0.719,18,55,False,0.902
XCR1,C_axis,0.0704,0.0807,0.872,0.387,18,55,False,0.701
CXCR3,C_axis,1.12,0.537,2.09,0.0409,18,55,False,0.264
CCR5,C_axis,-0.0161,0.291,-0.0552,0.956,18,55,False,0.973
CCR4,C_axis,-1.62,0.452,-3.59,0.000659,18,55,False,0.0382
CCR7,C_axis,-1.31,0.525,-2.49,0.0156,18,55,False,0.155
S1PR1,C_axis,-1.03,0.416,-2.48,0.0161,18,55,False,0.155
CX3CR1,C_axis,0.588,0.259,2.27,0.0265,18,55,False,0.22
XCL1,C_axis,0.529,0.685,0.773,0.443,18,55,False,0.748
XCL2,C_axis,0.236,0.653,0.361,0.719,18,55,False,0.902
CXCL9,C_axis,-0.206,0.129,-1.6,0.114,18,55,False,0.458
CXCL10,C_axis,-0.0583,0.261,-0.223,0.824,18,55,False,0.902
CXCL11,C_axis,-0.0104,0.0441,-0.236,0.814,18,55,False,0.902
CCL19,C_axis,0.0358,0.133,0.268,0.789,18,55,False,0.902
CCL21,C_axis,-0.139,0.0991,-1.4,0.166,18,55,False,0.508
IL12A,C_axis,0.0155,0.125,0.124,0.902,18,55,False,0.951
IL12B,C_axis,0.0224,0.015,1.5,0.139,18,55,False,0.458
IL18,C_axis,0.0434,0.143,0.303,0.763,18,55,False,0.902
IL18BP,C_axis,0.314,0.206,1.53,0.132,18,55,False,0.458
IL15,C_axis,-0.0655,0.157,-0.418,0.678,18,55,False,0.902
FLT3LG,C_axis,-0.191,0.314,-0.609,0.544,18,55,False,0.79
CD70,C_axis,0.543,0.36,1.51,0.136,18,55,False,0.458
TNFSF9,C_axis,0.0393,0.39,0.101,0.92,18,55,False,0.953
TNFSF4,C_axis,0.191,0.395,0.484,0.63,18,55,False,0.87
CD2,audit,0.548,0.513,1.07,0.29,18,55,False,0.686
CD3E,audit,-0.207,0.253,-0.816,0.418,18,55,False,0.734
PTPRC,audit,-0.175,0.118,-1.49,0.142,18,55,False,0.458
CD7,audit,-0.585,0.628,-0.932,0.355,18,55,False,0.686
GAPDH,audit,0.0961,0.101,0.948,0.349,6,37,True,0.686
```

Stage and cohort are confounded in this atlas. The crosstab below is printed in §6c-i; 12 of the 18 early paired donors are li2024, so no early-vs-advanced claim here is separable from a li2024-vs-rest claim, and the coefficient should be read as an upper bound on a stage effect, not as an estimate of one.

## §6c-i Stage x study, paired donors

**What:** The confound itself, as counts.

**How:** Independent paired donors only, from `target_cohort`'s `clin_stage_class`.

```csv
clin_stage_class,alkon2024,brentuximab2026,brunner2024,buus2025,chennareddy2025,gaydosik2019,gaydosik2022,herrera2021,il4ra2026,jonak2021,li2024,rindler2021_mc
advanced,0,6,5,0,5,3,5,4,9,0,19,4
early,5,0,0,0,0,0,0,0,0,1,16,0
unknown,0,0,0,3,0,0,0,0,0,0,0,0
```

## §6d Sensitivity -- depth-matched detection

**What:** Gate 1 detection recomputed after equalising sequencing depth, against the headline detection.

**How:** Counts are binomially thinned per cell to a common 2,000 UMI (seed 0); only cells at or above that depth take part (54.6% of T cells). `pct_expr` is then recomputed on the thinned counts and the same paired Wilcoxon is run. This is the direct answer to the depth difference between arms in §1b, and it is an addition to the brief.

**Table on disk:** `tables/pay51_sens_depth_matched.csv`

```csv
gene,family,log2fc_detect_depthmatched,q_detect_depthmatched,n_donors_paired_detect,log2fc_detect,q_detect,d_log2fc_detect
CD27,A_permitted,-0.269,0.949,75,-0.013,0.273,-0.256
CD40,A_permitted,-0.0111,0.0233,75,-0.0465,0.0147,0.0355
IFNGR1,A_permitted,-0.331,3.89e-05,75,0.0483,0.178,-0.379
IFNGR2,A_permitted,0.0475,0.227,75,0.46,0.000103,-0.413
IL10RA,A_permitted,-0.863,1.64e-10,75,-0.472,9.75e-05,-0.391
IL12RB1,A_permitted,-0.296,3.89e-05,75,0.146,0.0312,-0.442
IL12RB2,A_permitted,-0.337,2.36e-05,75,-0.212,0.00657,-0.125
IL15RA,A_permitted,-0.181,0.00847,75,0.0621,0.0434,-0.244
IL18R1,A_permitted,-0.372,5.99e-05,75,-0.0793,0.473,-0.292
IL18RAP,A_permitted,-0.349,0.000413,75,-0.257,0.0728,-0.0924
IL23R,A_permitted,0,0.949,75,0,0.503,0
IL2RB,A_permitted,-0.0461,0.635,75,0.327,0.000249,-0.373
IL2RG,A_permitted,0.0194,0.949,75,0.17,1.24e-05,-0.151
IL7R,A_permitted,-0.747,1.52e-10,75,-0.392,1.38e-05,-0.355
TNFRSF1A,A_permitted,-0.109,0.0626,75,0.164,0.00194,-0.272
TNFRSF1B,A_permitted,-0.24,0.00151,75,0.109,0.18,-0.35
TNFRSF9,A_permitted,-0.205,0.0281,75,-0.111,0.996,-0.0945
IL10RB,B_disqualified,0.0578,0.105,75,0.645,4.78e-10,-0.588
IL13RA1,B_disqualified,0,0.234,75,0.0306,0.00139,-0.0306
IL13RA2,B_disqualified,0,0.0122,75,0,0.00339,0
IL21R,B_disqualified,0.00827,0.541,75,0.534,1.55e-06,-0.525
IL27RA,B_disqualified,0.0378,0.82,75,0.479,9.81e-09,-0.441
IL2RA,B_disqualified,-0.271,0.0607,75,0.0171,0.141,-0.288
IL4R,B_disqualified,0.107,0.107,75,0.563,5.89e-08,-0.457
IL6R,B_disqualified,0.476,0.00809,75,0.778,9.81e-09,-0.302
IL9R,B_disqualified,0.282,0.0387,75,0.541,1.43e-05,-0.259
TNFRSF18,B_disqualified,0.0179,0.453,75,0.442,2.08e-05,-0.424
TNFRSF25,B_disqualified,0.489,0.0096,75,0.821,7.66e-08,-0.331
TNFRSF4,B_disqualified,0.317,0.0488,75,0.538,7.06e-06,-0.22
TNFRSF8,B_disqualified,0.164,0.0096,75,0.455,1.7e-07,-0.29
CCL19,C_axis,0,0.00212,75,0,0.853,0
CCL21,C_axis,0,0.0251,75,0,0.853,0
CCR4,C_axis,0.141,0.287,75,0.568,0.00138,-0.427
CCR5,C_axis,-0.213,0.00536,75,-0.279,0.00232,0.0658
CCR7,C_axis,-0.0744,0.406,75,0.197,0.018,-0.272
CD70,C_axis,0.61,8.08e-09,75,1.22,1.97e-12,-0.612
CX3CR1,C_axis,-0.0493,0.0149,75,-0.112,0.00327,0.0627
CXCL10,C_axis,0.0422,0.0178,75,0.137,7.51e-06,-0.0947
CXCL11,C_axis,0,0.829,75,0,0.0453,0
CXCL9,C_axis,0,0.0251,75,0,0.853,0
CXCR3,C_axis,-0.95,0.00356,75,-0.819,0.482,-0.131
FLT3LG,C_axis,-0.812,1.25e-08,75,-0.21,0.0362,-0.602
IL12A,C_axis,0,0.357,75,0,0.096,0
IL12B,C_axis,0,0.136,75,0,0.165,0
IL15,C_axis,0,0.822,75,0,0.35,0
IL18,C_axis,-0.153,8.08e-09,75,-0.112,1.58e-07,-0.0402
IL18BP,C_axis,-0.15,0.00633,75,-0.0609,0.853,-0.0894
S1PR1,C_axis,-0.789,6.11e-05,75,-0.596,0.0158,-0.193
TNFSF4,C_axis,0.038,0.0916,75,0.295,8.43e-06,-0.257
TNFSF9,C_axis,0.0931,0.115,75,0.485,1.17e-05,-0.392
XCL1,C_axis,-0.91,0.00212,75,-0.819,0.00805,-0.0912
XCL2,C_axis,-1.04,6.86e-05,75,-0.953,0.000407,-0.0889
XCR1,C_axis,0,0.488,75,0,0.156,0
CD2,audit,-0.192,0.000678,75,0.0676,0.975,-0.26
CD3E,audit,-0.0767,0.000678,75,0.0782,0.0487,-0.155
CD7,audit,-2.44,2.57e-11,75,-2.28,6e-11,-0.16
GAPDH,audit,0.00985,0.0076,45,0.0384,0.0432,-0.0285
PTPRC,audit,-0.0597,0.000453,75,0.0515,0.0826,-0.111
```

## §7 Contradictions and caveats

**What:** Everything in the data that undercuts a gate, named rather than left to the reader.

**How:** Assembled by hand from the diagnostics above; each row points at the section or the file that carries the number.

```csv
caveat,detail
tgt49 is not fixed by this job,"The (study, gene) coverage mask built here removes nothing from this 58-gene set, but the same mask applied to the 2,115-gene surface screen removes 342 genes from li2024 (15,790 genes in its source object), which is 31 of the 75 paired donors. tgt49's KIR3DL2 row (rank 522, log2FC 0, q 9.5e-4) is computed with those 31 donors contributing structural zeros to both arms. Those tables were left untouched by decision."
The healthy arm lost four donors,"SC320/SC348/SC353/SC354 are dropped because their count rows are empty, not because of anything biological. The counts exist in data/D14_gaydosik2023_skin/processed/standardized.h5ad and were not recovered."
The NK arm overlaps the T arms and is one study,NK-annotated cells are also labelled CD4 or CD8 in cell_type_T and remain inside those arms; and every one of them is li2024. XCL1 reach on NK is therefore a single-cohort number that is partly re-counted in the CD4 arms.
Detection is depth-confounded,"The malignant arm is sequenced deeper than the reactive arm (§1b), which inflates log2fc_detect and Gate 2 reach in the malignant direction. §6d is the matched version; where the two disagree, the matched one is the one to quote."
Expression and detection disagree for some genes,"log2fc_paired and log2fc_detect answer different questions -- how much per cell, versus in how many cells. Where they disagree (IL18R1/IL18RAP in the existing tables) both are reported and neither is chosen; a receptor that is detected in more cells at a lower level per cell is a different clinical object from one that is brighter on fewer."
Gate 2 and Gate 3 are descriptive,"No test is run on reach or on ligand pseudobulk. They carry no p value and no multiple testing correction, and a difference between arms there is an observation, not a result."
Myeloid subtypes come from a sidecar,"Mono_Mac/cDC1/cDC2/pDC labels are read from data/atlas_joint/skin_myeloid_fibro_subtypes.csv, assigned at cluster level with a re-clustering rescue step for the UNK cells; UNK is excluded here because some of it is T-cell doublets."
Family C mixes receptors and ligands in one BH correction,"Family C is corrected as one family of 23 genes as specified, although its receptors and its ligands are different hypotheses; the family column lets a reader re-correct the two subsets separately from pay51_cd4_paired.csv."
```
