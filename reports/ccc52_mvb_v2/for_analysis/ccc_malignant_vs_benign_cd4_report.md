# Differential cell–cell communication of malignant vs benign CD4 T cells (MF/SS atlas)

_Generated 2026-11-22 17:51 by `jobs/run_ccc52_mvb.py --annot v2`. Skin and blood are analysed completely separately._

> **Lineage: re-annotated (`nonmal_T_labels_v1`).** The benign arm is `CD4_conv` + `CD4_naive` in skin and `CD4_conv` in blood, from a scVI re-annotation of the non-malignant T/NK cells that the malignant cells never entered (`jobs/run_nonmal_T_reannot.py`). This supersedes `reports/ccc52_mvb/`, which used `cell_type_final == "CD4"` — an else-branch only ~45.8% conventional CD4 in skin. Cytotoxic and Treg genes are the ones that move; see §3.

## 1. Summary

**Skin.** Across **56 eligible donors** (178,977 malignant and 68,421 benign CD4 cells), the paired pseudobulk model tested 11,887 genes and 372 of the 1,043 ligand–receptor genes. **27 LR genes** meet all three call criteria — **9 up in the malignant arm** (PTHLH, PTH2R, CD70, CD9, CXCL13, PDGFA, IGFL2, KIR3DL2, …) and **18 down** (CCR6, SELL, CCL5, SORL1, CXCR4, FLT3LG, TNFSF8, CD55, …). 174 CD4-gene × cognate-gene × partner-type rows pass the 10% partner gate across 15 partner types.

**Blood.** Across **45 eligible donors** (217,023 malignant and 49,128 benign CD4 cells), the paired pseudobulk model tested 12,577 genes and 303 of the 1,043 ligand–receptor genes. **27 LR genes** meet all three call criteria — **17 up in the malignant arm** (PTHLH, KIR3DL2, TNFSF11, IGFL2, TNFSF9, CXCL13, CD70, IGFBP4, …) and **10 down** (CCL5, DPP4, ITGA4, LGALS3BP, SLC40A1, CXCR3, KLRG1, CD2, …). 163 CD4-gene × cognate-gene × partner-type rows pass the 10% partner gate across 13 partner types.

## 2. Rationale and estimand

Within a donor, malignant (M) and benign (B) CD4 cells face the same partner cells. For any multiplicative ligand–receptor score the partner term cancels in the M-vs-B contrast, so the differential interaction is exactly a **paired differential expression of ligand and receptor genes on the CD4 side**. Partner cells only act as a gate: is the cognate gene expressed in that cell type. The CD4-side result is the inference; the partner assignment is descriptive. Nothing here is an ambient correction, a LIANA specificity score, a latent-variable model, or a permutation test.

## 3. Data

### Paths used

| role | path |
|---|---|
| counts (raw integers, full gene space) | data/atlas_joint/joint_annotated.h5ad  [layers/raw_counts] |
| cell metadata | data/atlas_joint/atlas_obs_full_v2.parquet |
| malignancy call, skin | data/atlas_joint/skin_T_malignancy_cluster_v1.parquet |
| malignancy call, blood | data/atlas_joint/blood_T_malignancy_cluster_v1.parquet |
| lineage call for non-malignant T/NK (the benign arm) | data/atlas_joint/nonmal_T_labels_v1.parquet  [via helpers/lineage.py] |
| blood non-T fine labels | data/atlas_joint/blood_subtypes.csv |
| blood CD8→NK correction | data/atlas_joint/blood_T_celltype_corrections_v1.csv |
| LR resource (rebuilt by the `lr` stage) | reports/ccc52_mvb/lr_resource_1422.csv |

### obs columns used

| quantity | column |
|---|---|
| dataset / study | `dataset`, `study` |
| donor key | `donor` (dataset-prefixed, e.g. `D8__P78`; no cross-study collisions) |
| patient (unit of analysis) | `clin_person_key`, falling back to `patient_key`, then `donor` |
| sample | `sample_id` |
| tissue | `compartment` (Skin / Blood / LN) |
| malignancy | `mal_cluster` from the cluster-malignancy parquets |
| cell type | `cell_type_final` (19 levels; myeloid and fibroblast sub-clusters already collapsed) |
| chemistry | `tech` (10x_5p, 10x_flex, 10x_3p, ECCITE, 10x) |
| clinical | `clin_dx`, `clin_phenotype`, `clin_ex_vivo` |

### Cells per donor × tissue × arm

All 162 donor × tissue units that have at least one cell in either arm after the cohort rules. `eligible` marks the ones with >= 50 cells in **each** arm.

| tissue | unit | n_malignant | n_benign | eligible | study | tech | dx | n_samples | n_donor_keys |
|---|---|---|---|---|---|---|---|---|---|
| blood | washu:CTCL2 | 30525 | 0 | False | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:1084 | 22652 | 388 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 3 | 1 |
| blood | other/B7:SS3 | 19592 | 0 | False | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:1126 | 19304 | 1063 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 4 | 1 |
| blood | other/B7:SS1 | 15119 | 1539 | True | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P6 | 10753 | 857 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P3 | 8978 | 976 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P1 | 8831 | 361 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P8 | 8160 | 0 | False | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P4 | 7986 | 3540 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B6:P2 | 7976 | 3449 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B6:P9 | 7929 | 254 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | other/B7:SS2 | 7667 | 233 | True | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:999 | 7631 | 1010 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 1 | 1 |
| blood | other/B7:SS4 | 7500 | 514 | True | harro2023 | 10x_5p | SS | 1 | 1 |
| blood | washu:1135 | 7094 | 44 | False | dorando2026 | 10x_5p | MF_or_SS_unresolved | 2 | 1 |
| blood | pittsburgh:MF25 | 6752 | 240 | True | gaydosik2022 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P7 | 6535 | 1273 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | B1__P4 | 6187 | 6622 | True | borcherding2019 | 10x | SS | 4 | 2 |
| blood | B5__PT14 | 6145 | 470 | True | buus2025 | 10x_5p | SS | 2 | 1 |
| blood | B5__PT02 | 5610 | 73 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT06 | 5533 | 112 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P10 | 5426 | 1973 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B6:P5 | 5011 | 1702 | True | ren2023 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT05 | 4827 | 76 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT08 | 4408 | 119 | True | buus2025 | 10x_5p | SS | 2 | 1 |
| blood | B5__PT01 | 3294 | 7 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT09 | 3239 | 76 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | other/B6:P11 | 3112 | 687 | True | ren2023 | 10x_5p | MF | 1 | 1 |
| blood | B5__PT16 | 2952 | 219 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT18 | 2786 | 228 | True | buus2025 | 10x_5p | MF | 1 | 1 |
| blood | B5__PT07 | 2532 | 54 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT10 | 1781 | 59 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | H__SS1 | 1650 | 1083 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | H__MFIV1 | 1365 | 1188 | True | herrera2021 | ECCITE | MF | 1 | 1 |
| blood | H__SS6 | 1171 | 884 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | H__SS3 | 1042 | 319 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | pittsburgh:MF19 | 885 | 988 | True | gaydosik2022|herrera2021 | 10x_5p|ECCITE | SS | 2 | 2 |
| blood | H__SS5 | 863 | 368 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | washu:1181 | 781 | 8838 | True | dorando2026 | 10x_5p | MF_or_SS_unresolved | 3 | 1 |
| blood | B5__PT21 | 777 | 904 | True | buus2025 | 10x_5p | MF | 2 | 1 |
| blood | washu:CTCL6 | 557 | 636 | True | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT13 | 534 | 72 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT15 | 533 | 6 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | washu:CTCL5 | 524 | 177 | True | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | vienna:P303 | 447 | 628 | True | rindler2021_fi | 10x_5p | MF | 1 | 1 |
| blood | B5__PT04 | 434 | 23 | False | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | B5__PT23 | 342 | 1615 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | H__SS4 | 284 | 624 | True | herrera2021 | ECCITE | SS | 1 | 1 |
| blood | B5__PT22 | 266 | 556 | True | buus2025 | 10x_5p | SS | 1 | 1 |
| blood | pittsburgh:SZ16 | 260 | 1985 | True | gaydosik2022 | 10x_5p | SS | 1 | 1 |
| blood | washu:CTCL4 | 113 | 96 | True | borcherding2023 | 10x_5p | SS | 1 | 1 |
| blood | H__HC1 | 0 | 1368 | False | herrera2021 | ECCITE | HC | 1 | 1 |
| blood | other/B6:N1 | 0 | 8888 | False | ren2023 | 10x_5p | HC | 1 | 1 |
| blood | other/B6:N2 | 0 | 9541 | False | ren2023 | 10x_5p | HC | 1 | 1 |
| blood | other/B6:N3 | 0 | 8606 | False | ren2023 | 10x_5p | HC | 1 | 1 |
| blood | other/B7:MF1 | 0 | 36913 | False | harro2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B7:MF2 | 0 | 16922 | False | harro2023 | 10x_5p | MF | 1 | 1 |
| blood | other/B7:MF3 | 0 | 10884 | False | harro2023 | 10x_5p | MF | 1 | 1 |
| blood | pittsburgh:HB | 0 | 4017 | False | gaydosik2022 | 10x_5p | HC | 1 | 3 |
| blood | washu:1137 | 0 | 5411 | False | dorando2026 | 10x_5p | MF_or_SS_unresolved | 2 | 1 |
| blood | washu:CTCL3 | 0 | 1049 | False | borcherding2023 | 10x_5p | SS | 1 | 1 |
| skin | vienna:MF311 | 11646 | 1155 | True | chennareddy2025|lyp2026|rindler2021_mc | 10x_5p | MF | 4 | 3 |
| skin | B5__PT09 | 10427 | 1417 | True | buus2025 | 10x_5p | SS | 4 | 1 |
| skin | mda:MF28 | 10395 | 1023 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:MF309 | 10083 | 1532 | True | chennareddy2025|rindler2021_mc | 10x_5p | MF | 4 | 2 |
| skin | other/D11:P2 | 9473 | 548 | True | brentuximab2026 | 10x_5p | MF | 3 | 1 |
| skin | mda:CTCL3 | 8823 | 1108 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL1 | 7676 | 276 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | other/D11:P3 | 7596 | 566 | True | brentuximab2026 | 10x_5p | MF | 3 | 1 |
| skin | mda:CTCL5 | 7532 | 2913 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL8 | 7379 | 1747 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL18 | 7004 | 58 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | other/D11:P5 | 6550 | 735 | True | brentuximab2026 | 10x_5p | MF | 1 | 1 |
| skin | B5__PT23 | 6371 | 231 | True | buus2025 | 10x_5p | SS | 8 | 1 |
| skin | mda:CTCL13 | 5962 | 821 | True | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P171 | 5216 | 119 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | other/D11:P6 | 4575 | 298 | True | brentuximab2026 | 10x_5p | MF | 3 | 1 |
| skin | mda:CTCL6 | 3937 | 7812 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF17 | 3696 | 27 | False | gaydosik2022 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL7 | 3658 | 27708 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF48 | 3650 | 375 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT35 | 3502 | 42 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P204 | 3444 | 1926 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF26 | 3041 | 83 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P192 | 2874 | 158 | True | brunner2024 | 10x_5p | SS | 1 | 1 |
| skin | mda:CTCL12 | 2784 | 28 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P150 | 2393 | 476 | True | alkon2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF30 | 2342 | 1403 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL9 | 2216 | 48 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | pittsburgh:MF35 | 2187 | 291 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT55 | 2177 | 27 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:PGS | 2056 | 92 | True | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT11 | 1948 | 110 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P84 | 1892 | 1417 | True | rindler2021_mc | 10x_5p | MF | 2 | 1 |
| skin | mda:MF21 | 1761 | 172 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT50 | 1464 | 601 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT53 | 1417 | 209 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P65 | 1404 | 549 | True | alkon2024|rindler2021_mc | 10x_5p | MF | 2 | 2 |
| skin | mda:CTCL2 | 1357 | 1458 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT56 | 1292 | 1099 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL4 | 1241 | 2568 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL10 | 1151 | 44 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P303 | 1038 | 44 | False | chennareddy2025 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL16 | 1037 | 45 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:MF312 | 937 | 228 | True | chennareddy2025|rindler2021_mc | 10x_5p | MF | 2 | 2 |
| skin | vienna:P163 | 881 | 138 | True | alkon2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF18 | 826 | 426 | True | gaydosik2022 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:SC157 | 791 | 295 | True | gaydosik2019 | 10x_3p | MF | 1 | 1 |
| skin | other/D11:P1 | 678 | 23 | False | brentuximab2026 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:SC67 | 677 | 222 | True | gaydosik2019 | 10x_3p | SS | 1 | 1 |
| skin | pittsburgh:MF24 | 643 | 282 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P138 | 627 | 280 | True | alkon2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF36 | 620 | 322 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P196 | 506 | 46 | False | brunner2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF30 | 497 | 198 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P87 | 351 | 0 | False | brunner2024 | 10x_5p | SS | 1 | 1 |
| skin | vienna:JONAK1 | 348 | 950 | True | jonak2021 | 10x_5p | MF | 1 | 1 |
| skin | mda:PT47 | 307 | 86 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:SC158 | 306 | 11 | False | gaydosik2019 | 10x_3p | MF | 1 | 1 |
| skin | mda:MF22 | 300 | 208 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF19 | 282 | 88 | True | gaydosik2022|herrera2021 | 10x_5p|ECCITE | SS | 2 | 2 |
| skin | pittsburgh:MF34 | 248 | 344 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | H__SS4 | 204 | 25 | False | herrera2021 | ECCITE | SS | 1 | 1 |
| skin | mda:CTCL17 | 203 | 9 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | vienna:P73 | 189 | 231 | True | rindler2021_mc | 10x_5p | MF | 2 | 1 |
| skin | other/D11:P4 | 174 | 233 | True | brentuximab2026 | 10x_5p | MF | 2 | 1 |
| skin | mda:PT52 | 160 | 48 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P90 | 126 | 49 | False | alkon2024|rindler2021_mc | 10x_5p | MF | 2 | 2 |
| skin | pittsburgh:MF28 | 123 | 298 | True | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF21 | 118 | 79 | True | gaydosik2022 | 10x_5p | MF | 1 | 1 |
| skin | vienna:P105 | 117 | 164 | True | brunner2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF27 | 107 | 202 | True | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:CTCL15 | 103 | 0 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | pittsburgh:MF25 | 99 | 93 | True | gaydosik2022 | 10x_5p | MF/SS | 1 | 1 |
| skin | mda:CTCL11 | 90 | 0 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | mda:CTCL14 | 63 | 26 | False | li2024 | 10x_flex | MF | 1 | 1 |
| skin | H__MFIV1 | 44 | 9 | False | herrera2021 | ECCITE | MF | 1 | 1 |
| skin | mda:MF17 | 37 | 3185 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | H__SS1 | 36 | 52 | False | herrera2021 | ECCITE | SS | 1 | 1 |
| skin | H__SS3 | 19 | 32 | False | herrera2021 | ECCITE | SS | 1 | 1 |
| skin | B5__PT22 | 10 | 3053 | False | buus2025 | 10x_5p | SS | 8 | 1 |
| skin | mda:MF15 | 0 | 954 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF18 | 0 | 134 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | mda:MF26 | 0 | 1634 | False | li2024 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:MF40 | 0 | 105 | False | il4ra2026 | 10x_5p | MF | 1 | 1 |
| skin | pittsburgh:NS13 | 0 | 125 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:NS14 | 0 | 83 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:NS16 | 0 | 28 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:NS17 | 0 | 140 | False | il4ra2026 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:PF2 | 0 | 582 | False | gaydosik2022 | 10x_5p | HC | 1 | 2 |
| skin | pittsburgh:SC124 | 0 | 35 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | pittsburgh:SC125 | 0 | 31 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | pittsburgh:SC327 | 0 | 135 | False | gaydosik2022 | 10x_5p | HC | 1 | 2 |
| skin | pittsburgh:SC405 | 0 | 21 | False | gaydosik2023 | 10x_5p | HC | 1 | 1 |
| skin | pittsburgh:SC50 | 0 | 21 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | pittsburgh:SC68 | 0 | 22 | False | gaydosik2019 | 10x_3p | HC | 1 | 1 |
| skin | vienna:P112 | 0 | 132 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P115 | 0 | 248 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P116 | 0 | 196 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P121 | 0 | 33 | False | chennareddy2025 | 10x_5p | HC | 1 | 1 |
| skin | vienna:P94 | 0 | 2088 | False | brunner2024 | 10x_5p | MF | 1 | 1 |

### Eligible donors per dataset and tissue

| tissue | study | n_eligible_donors |
|---|---|---|
| blood | buus2025 | 14 |
| blood | ren2023 | 10 |
| blood | herrera2021 | 6 |
| blood | dorando2026 | 4 |
| blood | borcherding2023 | 3 |
| blood | harro2023 | 3 |
| blood | gaydosik2022 | 2 |
| blood | borcherding2019 | 1 |
| blood | gaydosik2022|herrera2021 | 1 |
| blood | rindler2021_fi | 1 |
| skin | li2024 | 20 |
| skin | il4ra2026 | 8 |
| skin | brentuximab2026 | 5 |
| skin | alkon2024 | 3 |
| skin | chennareddy2025 | 3 |
| skin | gaydosik2022 | 3 |
| skin | brunner2024 | 2 |
| skin | buus2025 | 2 |
| skin | chennareddy2025|rindler2021_mc | 2 |
| skin | gaydosik2019 | 2 |
| skin | rindler2021_mc | 2 |
| skin | alkon2024|rindler2021_mc | 1 |
| skin | chennareddy2025|lyp2026|rindler2021_mc | 1 |
| skin | gaydosik2022|herrera2021 | 1 |
| skin | jonak2021 | 1 |

### Donors excluded, and why

| tissue | drop_reason | n_units | n_cells |
|---|---|---|---|
| blood | ex vivo culture cohort (geskin2026) | 11 | 260455 |
| skin | diagnosis outside MF/SS | 15 | 140974 |
| skin | malignant clone is not CD4 | 11 | 136574 |

Units that survived the cohort rules but failed the 50-cells-per-arm floor: skin 44, blood 17.

### Benign-arm definition (re-annotated lineage)

The benign arm is **not** `cell_type_final == "CD4"`. That label is an else-branch: nb21 and nb31 hand-mapped Leiden clusters off dot plots and gave `CD4` to every T cluster not positively called something else, and in blood the `CD4` transcript was not even in the object when the call was made. `reports/ccc52_mvb/qc_benign_arm/` measured the result — 21.9% of the skin benign arm sat in clusters with CD8A/B detection >= 0.25, with a Treg admixture on top, and those are real cells rather than ambient or doublets, so no expression filter could remove them.

`jobs/run_nonmal_T_reannot.py` rebuilt the call from a scVI embedding of the non-malignant T/NK cells that the malignant cells never entered, and wrote a label for all 576,188 of them. `helpers/lineage.py` is the only reader. The arm here is **`CD4_conv` + `CD4_naive` in skin and `CD4_conv` in blood** — the same naive convention v1 had, where the blood clean-up dropped naive clusters and skin kept them. `Mixed`, `LowQC` and `Suspect_malignant` never enter an arm. The malignant arm is unchanged: malignant cells are absent from the re-annotation by construction, so they still carry the old `CD4`, and the script asserts that none of them was relabelled.

No hand-built blood clean-up is needed any more — the re-annotation supplies `CD4_Treg` and `CD4_naive` in both tissues, which is what that step was approximating with a FOXP3/CCR7/SELL/TCF7/LEF1 detection rule on per-donor clusters.

Net effect on the benign arm, over cohort-eligible cells:

| tissue | old benign CD4 arm | kept | removed by the re-annotation | admitted from another v1 label | new benign arm |
|---|---|---|---|---|---|
| skin | 154391 | 76872 | 77519 | 5169 | 82041 |
| blood | 148638 | 116595 | 32043 | 36212 | 152807 |

What the removed cells turned out to be (old `CD4`, non-malignant, that no longer reach the benign arm, by their new label):

| tissue | label_v2 | n_cells_removed |
|---|---|---|
| blood | CD4_Treg | 16423 |
| blood | CD4_naive | 6374 |
| blood | CD8 | 5183 |
| blood | Mixed | 2398 |
| blood | LowQC | 1022 |
| blood | NK_ILC | 334 |
| blood | MAIT | 159 |
| blood | Prolif | 111 |
| blood | Suspect_malignant | 34 |
| blood | gdT | 5 |
| skin | CD8 | 23351 |
| skin | Mixed | 19050 |
| skin | CD4_Treg | 18126 |
| skin | NK_ILC | 11420 |
| skin | Prolif | 3529 |
| skin | Suspect_malignant | 1213 |
| skin | LowQC | 450 |
| skin | gdT | 380 |

Per old label × new label × malignancy, the whole table is `arm_composition.csv`.

#### What this changed in the result

Every gene fitted under both annotations, by how far its effect size moved:

| verdict | blood | skin |
|---|---|---|
| moved (0.5-1.5) | 40 | 79 |
| stable (|shift| < 0.5) | 263 | 287 |
| v1 was driven by the arm (|shift| >= 1.5) | 5 | 23 |

The genes the benign-arm contamination was inflating, and the controls that should not move:

| tissue | gene | log2FC_v1 | log2FC_v2 | shift_log2FC | pct_B_v1 | pct_B_v2 |
|---|---|---|---|---|---|---|
| blood | CD8A | -5.52 | -5.87 | -0.35 | 0.06 | 0.03 |
| blood | CD8B | -6.42 | -5.92 | 0.50 | 0.06 | 0.02 |
| blood | KLRD1 | -3.45 | -5.17 | -1.72 | 0.02 | 0.02 |
| blood | TRDC | -6.14 | -6.86 | -0.73 | 0.01 | 0.01 |
| blood | NKG7 | -3.55 | -6.52 | -2.98 | 0.08 | 0.13 |
| blood | GZMK | -5.60 | -5.73 | -0.13 | 0.09 | 0.09 |
| blood | FOXP3 | -1.72 | -0.12 | 1.61 | 0.04 | 0.02 |
| blood | TOX | 2.72 | 2.93 | 0.21 | 0.09 | 0.06 |
| blood | CD70 | 2.62 | 2.96 | 0.34 | 0.03 | 0.02 |
| blood | KIR3DL2 | 3.92 | 4.66 | 0.74 | 0.02 | 0.01 |
| blood | CXCR4 | -0.56 | -0.77 | -0.21 | 0.54 | 0.59 |
| blood | CCR4 | 1.09 | 1.57 | 0.48 | 0.11 | 0.06 |
| skin | CD8A | -4.74 | -3.47 | 1.27 | 0.17 | 0.08 |
| skin | CD8B | -5.16 | -3.68 | 1.48 | 0.05 | 0.02 |
| skin | KLRD1 | -6.02 | -2.41 | 3.61 | 0.07 | 0.01 |
| skin | TRDC | -5.71 | -2.41 | 3.30 | 0.03 | 0.01 |
| skin | NKG7 | -3.77 | -0.87 | 2.90 | 0.20 | 0.08 |
| skin | GZMK | -4.51 | -3.90 | 0.61 | 0.10 | 0.06 |
| skin | FOXP3 | -3.94 | -2.55 | 1.39 | 0.09 | 0.05 |
| skin | TOX | 1.16 | 1.15 | -0.00 | 0.12 | 0.09 |
| skin | CD70 | 1.20 | 1.96 | 0.76 | 0.05 | 0.03 |
| skin | KIR3DL2 | 0.49 | 1.01 | 0.52 | 0.01 | 0.01 |
| skin | CXCR4 | -1.45 | -1.77 | -0.32 | 0.69 | 0.72 |
| skin | CCR4 | -0.18 | -0.48 | -0.29 | 0.21 | 0.25 |

Read this as two different stories. **In skin the contamination was inflating the cytotoxic and Treg effects, and removing it deflates them** — exactly the failure the re-annotation was built to fix. **In blood several cytotoxic effects get larger, not smaller**, because the v1 blood arm was a flat `CD4` with naive and Treg clusters stripped out by a marker rule, while the v2 arm is `CD4_conv` proper; its benign cells carry more NKG7 and CCL5 (`pct_B` rises), so the malignant-vs-benign gap widens. Neither direction is evidence about the malignant clone — both are the benign arm changing underneath it.

Full table: `annotation_shift.csv`. **A v1 number whose verdict is not `stable` should not be quoted.**

#### The arm is defined by genes, so those genes cannot be read as biology

The re-annotation routes a cell into the benign arm using the detection of a fixed marker panel, so the M-vs-B contrast of those **28 genes is partly circular**: a cell high in CD8B left the benign arm *because* it was high in CD8B. Every table here carries a `label_defining_gene` column; the panel is

`CCR7`, `CD3D`, `CD3E`, `CD4`, `CD40LG`, `CD8A`, `CD8B`, `CTLA4`, `FCGR3A`, `FOXP3`, `GZMK`, `IKZF2`, `IL2RA`, `IL7R`, `KIT`, `KLRB1`, `KLRF1`, `LEF1`, `NCAM1`, `NKG7`, `S100A4`, `SELL`, `SLC4A10`, `TCF7`, `TRDC`, `TRGC1`, `TRGC2`, `TYROBP`

Of these, significant under v2: `CTLA4` (blood, log2FC +1.46), `CD40LG` (skin, log2FC -1.10), `CD4` (skin, log2FC -1.39), `SELL` (skin, log2FC -2.42).

In skin `CD4` and `CD40LG` are significant and were not under v1 — that is the selection, not a discovery about the clone. v1 was not free of this either: nb21 and nb31 hand-mapped clusters off dot plots of the same markers. The difference is that v2 makes the rule explicit, so the affected genes can be named. Genes outside the panel — `CD70`, `KIR3DL2`, `TOX`, `PTHLH`, `CXCL13`, `DPP4`, `CCR4` — are not touched by it.

### Partner cell types

A coarse annotation per tissue, taken from the re-annotated lineage for T/NK cells and `cell_type_final` for everything else (myeloid, fibroblast, keratinocyte, B, ...), so both tissues now carry real `CD4_Treg`, `CD4_naive`, `CD8`, `NK_ILC`, `gdT` and `MAIT` partners and the nb32b CD8->NK hand correction is applied only where the re-annotation has no opinion, with the malignant and benign CD4 arms added as types of their own. Pooled over eligible donors of that tissue; types below 200 cells are dropped.

| tissue | partner | n_cells | homotypic |
|---|---|---|---|
| blood | CD4_malignant | 217023 | True |
| blood | CD4_benign | 49128 | False |
| blood | Mono_CD14 | 20402 | False |
| blood | CD8 | 11837 | False |
| blood | B | 8572 | False |
| blood | NK_ILC | 8414 | False |
| blood | CD4_Treg | 7171 | False |
| blood | Mono_CD16 | 3030 | False |
| blood | CD4_naive | 2681 | False |
| blood | Platelet | 2202 | False |
| blood | Plasma | 1066 | False |
| blood | Erythroid | 505 | False |
| blood | cDC | 460 | False |
| skin | CD4_malignant | 178977 | True |
| skin | Keratinocyte | 171452 | False |
| skin | Fibroblast | 96967 | False |
| skin | Myeloid | 82371 | False |
| skin | CD4_benign | 68421 | False |
| skin | Vascular | 61275 | False |
| skin | CD8 | 53388 | False |
| skin | CD4_Treg | 46853 | False |
| skin | B | 24188 | False |
| skin | Melanocyte | 10332 | False |
| skin | Plasma | 10185 | False |
| skin | NK_ILC | 10030 | False |
| skin | Prolif | 3292 | False |
| skin | Mast | 2784 | False |
| skin | gdT | 750 | False |

## 4. Methods

**Counts.** Raw integer counts were read from `data/atlas_joint/joint_annotated.h5ad`, layer `raw_counts` (2,157,693 × 42,347, CSR). Integer-ness and non-negativity are asserted on the first streamed chunk. The compartment objects were deliberately not used for counts: `blood_T_annotated.h5ad` is a 10,000-HVG subset that holds only 590 of the 1,043 LR genes, and neither compartment object contains non-T partner cells.

**Arms.** Malignancy is `mal_cluster`, the repo headline call since 2026-10-22 (`helpers/malignancy.py`): per-sample Leiden over all T cells with one call per cluster, anchored per sample on the ALICE TCR clone. M = CD4 T cells called malignant; B = CD4 T cells called non-malignant. `cell_type_final == "CD4"` already excludes `CD4_Treg`, which is a separate level. Cells with a null (unassessed) call enter neither arm.

**Unit of analysis.** The donor, defined as `clin_person_key` (the repo's curated person key, which merges the same patient across centres) falling back to `patient_key` and then `donor`. All samples and all timepoints of a patient within a tissue are summed into one pseudobulk per arm.

**Cohort rules.** Applied before summation, following `helpers/target_cohort.py`: skin donors whose malignant clone is not CD4 (`clin_phenotype` in CD8/gd/DN, or `clin_dx` in CD8_aggr_CTCL/pcAECTCL) are dropped, because their CD4 cells sit in the benign arm beside a tumour that is a different lineage; skin donors with a diagnosis outside the MF/SS spectrum (LyP, SPP, LPP) are dropped; the whole geskin2026 blood cohort is dropped. The 'same person under two donor keys' rule is not applied as a drop — patient-level aggregation already handles it by summing.

**Eligibility.** A donor × tissue needs >= 50 cells in each arm.

**Pseudobulk.** Raw counts summed over all cells of a donor × arm within a tissue, in one streamed pass over the atlas in chunks of 20,000 cells. The same pass accumulates per-group detection counts (cells with count > 0) and per-chemistry detection counts.

**LR resource.** LIANA `consensus` intersected with interactions supported by >= 6 of the 15 human LIANA resources = **1,422 interactions over 1,043 unique genes** (asserted). Complexes are underscore-joined and each subunit is treated as a gene. No blacklist was applied — see §10.

**Gene filter.** A gene enters a fit if it has >= 10 counts in >= 25% of that fit's pseudobulks.

**Model.** pydeseq2, paired pseudobulk, exact formula **`~ donor + arm`**, reference level `B`, contrast `arm M vs B`, fitted on the full transcriptome (not only LR genes). `refit_cooks=True`. Shrunken log2 fold changes come from `DeseqStats.lfc_shrink` on the `arm` coefficient; the unshrunken estimate is retained as `log2FC_raw`.

**Mixed chemistries.** A 10x Flex probe panel leaves off-panel genes at exactly zero in every cell of that chemistry. Where a tissue mixes chemistries, fit (i) uses **all** eligible donors over the genes detected in every chemistry present, and fit (ii) uses **non-Flex donors only** for every remaining expressed gene. The two gene sets are disjoint and together cover every gene that passes the count filter, so no gene is stranded by a small chemistry that happens to miss it. The `fit` column records which fit each gene came from. Flex (the only probe-panel chemistry here) is li2024 and skin-only, so in blood fit (ii) keeps every donor and is labelled `chemistry-restricted` rather than `non-Flex`. The chemistries actually present per tissue are: **skin** 10x_3p, 10x_5p, 10x_5p|ECCITE, 10x_flex; **blood** 10x, 10x_5p, 10x_5p|ECCITE, ECCITE.

**Multiple testing.** Benjamini–Hochberg applied to the DESeq2 Wald p-values **only over the genes present in the filtered LR resource**, once per tissue per fit. Genes outside the resource carry a p-value and no padj. DESeq2's own independent-filtering padj is not used, because its family is the whole transcriptome.

**Call.** padj < 0.05, |shrunken log2FC| >= 1.0, and the gene detected in >= 10% of the cells of the higher arm (pooled over eligible donors of that tissue).

**Donor concordance.** Per-donor log2FC from CPM with pseudocount 1, `log2((CPM_M + 1)/(CPM_B + 1))`; the reported count is the number of donors whose sign matches the model estimate.

**Complexes.** Every subunit is evaluated as a gene. A complex is called on the CD4 side through its significant subunit provided every other subunit is detected in >= 10% of the cells of that arm (`cd4_complex_ok`).

**Partner gate.** The cognate gene — all subunits, for a complex — must be detected in >= 10% of the cells of the partner type, pooled over eligible donors of that tissue. No statistical test is applied on the partner side. `outgoing` = CD4 ligand to partner receptor; `incoming` = partner ligand to CD4 receptor.

**Reproducibility.** Seed 0. Versions: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, pydeseq2 0.5.4, anndata 0.12.7, h5py 3.15.1, matplotlib 3.10.8, polars 1.40.1, python 3.12.12.

## 5. Results, skin

Genes tested: **11,887** over 2 fit(s) (all-chemistry, non-Flex); of those **372** are in the filtered LR resource and carry a BH adjusted p-value.

Significant at padj < 0.05, |shrunken log2FC| >= 1.0, and >= 10% detection in the higher arm: **27** genes — **9 up in malignant**, **18 down**. By role: ligand 12, receptor 15, both 0 (up: L 5 / R 4; down: L 7 / R 11).

**Top 9 up in malignant** (by shrunken log2FC):

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| PTHLH | ligand | 116.373 | 2.190 | 2.97e-11 | 0.150 | 0.010 | 48 | 56 | all-chemistry |
| PTH2R | receptor | 64.080 | 2.009 | 0.003 | 0.166 | 0.006 | 38 | 56 | all-chemistry |
| CD70 | ligand | 213.142 | 1.961 | 3.92e-23 | 0.223 | 0.030 | 51 | 56 | all-chemistry |
| CD9 | receptor | 279.898 | 1.839 | 8.76e-12 | 0.316 | 0.025 | 47 | 56 | all-chemistry |
| CXCL13 | ligand | 1943.867 | 1.734 | 4.04e-06 | 0.253 | 0.032 | 40 | 56 | all-chemistry |
| PDGFA | ligand | 72.169 | 1.712 | 4.09e-11 | 0.110 | 0.010 | 45 | 56 | all-chemistry |
| IGFL2 | ligand | 224.645 | 1.597 | 9.78e-06 | 0.192 | 0.018 | 35 | 56 | all-chemistry |
| KIR3DL2 | receptor | 110.466 | 1.008 | 0.004 | 0.175 | 0.009 | 22 | 56 | non-Flex |
| CD151 | receptor | 142.708 | 1.001 | 9.67e-15 | 0.251 | 0.055 | 46 | 56 | all-chemistry |

**Top 18 down in malignant**:

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| CCR6 | receptor | 203.104 | -3.091 | 4.40e-24 | 0.089 | 0.123 | 42 | 56 | all-chemistry |
| SELL | receptor | 588.934 | -2.417 | 9.25e-16 | 0.220 | 0.298 | 41 | 56 | all-chemistry |
| CCL5 | ligand | 671.530 | -2.307 | 4.67e-21 | 0.153 | 0.125 | 48 | 56 | all-chemistry |
| SORL1 | receptor | 199.606 | -1.875 | 9.91e-11 | 0.172 | 0.197 | 42 | 56 | all-chemistry |
| CXCR4 | receptor | 2971.526 | -1.769 | 2.00e-42 | 0.545 | 0.722 | 55 | 56 | all-chemistry |
| FLT3LG | ligand | 376.453 | -1.502 | 1.50e-33 | 0.238 | 0.253 | 50 | 56 | all-chemistry |
| TNFSF8 | ligand | 179.831 | -1.497 | 6.41e-09 | 0.146 | 0.131 | 40 | 56 | all-chemistry |
| CD55 | ligand | 525.375 | -1.399 | 1.05e-15 | 0.287 | 0.336 | 48 | 56 | all-chemistry |
| CD4 | receptor | 245.351 | -1.394 | 4.88e-08 | 0.182 | 0.168 | 46 | 56 | all-chemistry |
| LTB | ligand | 2363.668 | -1.345 | 4.99e-12 | 0.545 | 0.654 | 44 | 56 | all-chemistry |
| CD5 | receptor | 463.667 | -1.213 | 3.63e-07 | 0.271 | 0.411 | 39 | 56 | all-chemistry |
| ICOS | receptor | 461.107 | -1.209 | 3.87e-07 | 0.279 | 0.222 | 41 | 56 | all-chemistry |
| IL10RA | receptor | 303.962 | -1.187 | 1.16e-16 | 0.236 | 0.258 | 48 | 56 | all-chemistry |
| FAS | receptor | 189.591 | -1.162 | 3.21e-09 | 0.158 | 0.135 | 37 | 56 | all-chemistry |
| CD40LG | ligand | 380.527 | -1.103 | 1.21e-06 | 0.255 | 0.225 | 39 | 56 | all-chemistry |
| CD28 | receptor | 374.532 | -1.044 | 9.07e-06 | 0.309 | 0.274 | 36 | 56 | all-chemistry |
| KCNA3 | receptor | 103.144 | -1.035 | 1.30e-14 | 0.115 | 0.112 | 46 | 56 | all-chemistry |
| AREG | ligand | 664.066 | -1.025 | 6.76e-05 | 0.194 | 0.115 | 36 | 56 | all-chemistry |

**Gated interactions for those genes** — 174 CD4-gene x cognate x partner rows pass the 10% partner gate (22 CD4-side genes, 15 partner types). The full set for every significant gene is in `interactions_skin.csv` (174 rows).

| cd4_gene | sign | direction | cognate_gene | partner_type | partner_pct | partner_mean_cp10k |
|---|---|---|---|---|---|---|
| AREG | lost in M | outgoing | ERBB3 | Melanocyte | 0.316 | 0.940 |
| AREG | lost in M | outgoing | ERBB3 | Keratinocyte | 0.221 | 0.751 |
| CCL5 | lost in M | outgoing | ACKR1 | Vascular | 0.487 | 13.615 |
| CCL5 | lost in M | outgoing | SDC1 | Keratinocyte | 0.386 | 2.004 |
| CCL5 | lost in M | outgoing | CXCR3 | Prolif | 0.370 | 1.332 |
| CCL5 | lost in M | outgoing | SDC4 | Keratinocyte | 0.343 | 1.468 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_Treg | 0.299 | 2.090 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_malignant | 0.266 | 1.264 |
| CCL5 | lost in M | outgoing | SDC1 | Plasma | 0.262 | 2.085 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_benign | 0.248 | 2.042 |
| CCL5 | lost in M | outgoing | CCR4 | Prolif | 0.226 | 0.843 |
| CCL5 | lost in M | outgoing | CXCR3 | CD8 | 0.217 | 1.777 |
| CCL5 | lost in M | outgoing | SDC4 | Prolif | 0.202 | 0.742 |
| CCL5 | lost in M | outgoing | CXCR3 | gdT | 0.188 | 1.413 |
| CCL5 | lost in M | outgoing | CXCR3 | Plasma | 0.173 | 0.763 |
| CCL5 | lost in M | outgoing | CXCR3 | NK_ILC | 0.171 | 1.027 |
| CCL5 | lost in M | outgoing | ACKR4 | Fibroblast | 0.163 | 0.629 |
| CCL5 | lost in M | outgoing | SDC4 | Fibroblast | 0.158 | 0.493 |
| CCL5 | lost in M | outgoing | SDC4 | Myeloid | 0.156 | 0.425 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_Treg | 0.146 | 0.908 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_malignant | 0.117 | 0.516 |
| CCL5 | lost in M | outgoing | CCR4 | CD8 | 0.110 | 0.777 |
| CD28 | lost in M | incoming | CD86 | Myeloid | 0.463 | 2.818 |
| CD28 | lost in M | incoming | CD80 | Myeloid | 0.190 | 0.691 |
| CD4 | lost in M | incoming | CXCL12 | Fibroblast | 0.615 | 11.451 |
| CD4 | lost in M | incoming | IL16 | B | 0.404 | 2.695 |
| CD4 | lost in M | incoming | IL16 | CD4_malignant | 0.361 | 1.642 |
| CD4 | lost in M | incoming | IL16 | gdT | 0.343 | 2.311 |
| CD4 | lost in M | incoming | IL16 | Prolif | 0.341 | 0.934 |
| CD4 | lost in M | incoming | IL16 | NK_ILC | 0.329 | 1.952 |
| CD4 | lost in M | incoming | IL16 | CD4_benign | 0.328 | 2.577 |
| CD4 | lost in M | incoming | IL16 | CD4_Treg | 0.310 | 1.930 |
| CD4 | lost in M | incoming | IL16 | CD8 | 0.307 | 2.189 |
| CD4 | lost in M | incoming | IL16 | Plasma | 0.262 | 0.949 |
| CD4 | lost in M | incoming | CXCL12 | Vascular | 0.240 | 2.249 |
| CD4 | lost in M | incoming | IL16 | Melanocyte | 0.194 | 0.610 |
| CD4 | lost in M | incoming | IL16 | Myeloid | 0.146 | 0.373 |
| CD4 | lost in M | incoming | IL16 | Fibroblast | 0.118 | 0.331 |
| CD40LG | lost in M | outgoing | CD40 | B | 0.474 | 3.719 |
| CD40LG | lost in M | outgoing | CD40 | Myeloid | 0.431 | 2.964 |
| CD40LG | lost in M | outgoing | CD40 | Vascular | 0.344 | 1.255 |
| CD40LG | lost in M | outgoing | CD40 | Fibroblast | 0.220 | 0.898 |
| CD40LG | lost in M | outgoing | CD40 | Plasma | 0.166 | 0.537 |
| CD40LG | lost in M | outgoing | CD40 | Keratinocyte | 0.126 | 0.303 |
| CD55 | lost in M | outgoing | ADGRE5 | NK_ILC | 0.502 | 4.736 |
| CD55 | lost in M | outgoing | ADGRE5 | Prolif | 0.477 | 1.956 |
| CD55 | lost in M | outgoing | ADGRE5 | CD4_malignant | 0.333 | 1.655 |
| CD55 | lost in M | outgoing | ADGRE5 | CD8 | 0.269 | 2.278 |
| CD55 | lost in M | outgoing | ADGRE5 | Plasma | 0.252 | 0.994 |
| CD55 | lost in M | outgoing | ADGRE5 | Myeloid | 0.252 | 0.791 |
| CD55 | lost in M | outgoing | ADGRE5 | Fibroblast | 0.241 | 0.792 |
| CD55 | lost in M | outgoing | ADGRE5 | B | 0.227 | 1.657 |
| CD55 | lost in M | outgoing | ADGRE5 | CD4_Treg | 0.213 | 1.177 |
| CD55 | lost in M | outgoing | ADGRE5 | gdT | 0.209 | 1.380 |
| CD55 | lost in M | outgoing | ADGRE5 | CD4_benign | 0.154 | 1.127 |
| CD55 | lost in M | outgoing | ADGRE5 | Mast | 0.147 | 1.164 |
| CD55 | lost in M | outgoing | ADGRE5 | Melanocyte | 0.114 | 0.283 |
| CD70 | gained in M | outgoing | CD27 | CD4_Treg | 0.550 | 5.032 |
| CD70 | gained in M | outgoing | CD27 | Prolif | 0.512 | 2.669 |
| CD70 | gained in M | outgoing | CD27 | Plasma | 0.450 | 4.382 |
| CD70 | gained in M | outgoing | CD27 | CD8 | 0.402 | 4.216 |
| CD70 | gained in M | outgoing | CD27 | CD4_malignant | 0.370 | 2.537 |
| CD70 | gained in M | outgoing | CD27 | gdT | 0.248 | 1.961 |
| CD70 | gained in M | outgoing | CD27 | CD4_benign | 0.228 | 1.891 |
| CD70 | gained in M | outgoing | CD27 | B | 0.160 | 1.239 |
| CD70 | gained in M | outgoing | CD27 | NK_ILC | 0.106 | 0.588 |
| CD9 | gained in M | incoming | HBEGF | Keratinocyte | 0.255 | 1.508 |
| CD9 | gained in M | incoming | HBEGF | Vascular | 0.238 | 1.059 |
| CD9 | gained in M | incoming | HBEGF | Myeloid | 0.141 | 0.610 |
| CXCL13 | gained in M | outgoing | CXCR3 | Prolif | 0.370 | 1.332 |
| CXCL13 | gained in M | outgoing | CXCR5 | B | 0.287 | 2.130 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD8 | 0.217 | 1.777 |
| CXCL13 | gained in M | outgoing | CXCR3 | gdT | 0.188 | 1.413 |
| CXCL13 | gained in M | outgoing | CXCR3 | Plasma | 0.173 | 0.763 |
| CXCL13 | gained in M | outgoing | CXCR3 | NK_ILC | 0.171 | 1.027 |
| CXCL13 | gained in M | outgoing | ACKR4 | Fibroblast | 0.163 | 0.629 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_Treg | 0.146 | 0.908 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_malignant | 0.117 | 0.516 |
| CXCL13 | gained in M | outgoing | CCR10 | Prolif | 0.101 | 0.292 |
| CXCR4 | lost in M | incoming | CXCL12 | Fibroblast | 0.615 | 11.451 |

## 6. Results, blood

Genes tested: **12,577** over 2 fit(s) (all-chemistry, chemistry-restricted); of those **303** are in the filtered LR resource and carry a BH adjusted p-value.

Significant at padj < 0.05, |shrunken log2FC| >= 1.0, and >= 10% detection in the higher arm: **27** genes — **17 up in malignant**, **10 down**. By role: ligand 10, receptor 17, both 0 (up: L 7 / R 10; down: L 3 / R 7).

**Top 17 up in malignant** (by shrunken log2FC):

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| PTHLH | ligand | 92.847 | 5.461 | 2.79e-41 | 0.167 | 0.006 | 45 | 45 | all-chemistry |
| KIR3DL2 | receptor | 362.215 | 4.660 | 1.60e-41 | 0.369 | 0.010 | 41 | 45 | all-chemistry |
| TNFSF11 | ligand | 83.077 | 4.319 | 5.56e-45 | 0.120 | 0.006 | 45 | 45 | all-chemistry |
| IGFL2 | ligand | 169.014 | 3.934 | 1.96e-11 | 0.113 | 0.007 | 39 | 45 | all-chemistry |
| TNFSF9 | ligand | 120.799 | 3.379 | 9.42e-34 | 0.175 | 0.010 | 42 | 45 | all-chemistry |
| CXCL13 | ligand | 169.501 | 3.375 | 7.13e-11 | 0.137 | 0.005 | 36 | 45 | all-chemistry |
| CD70 | ligand | 234.493 | 2.960 | 4.22e-31 | 0.212 | 0.023 | 41 | 45 | all-chemistry |
| IGFBP4 | ligand | 282.748 | 2.684 | 2.23e-30 | 0.218 | 0.036 | 42 | 45 | all-chemistry |
| FGFR1 | receptor | 75.500 | 2.269 | 1.14e-15 | 0.100 | 0.019 | 41 | 45 | all-chemistry |
| CCR4 | receptor | 274.822 | 1.566 | 9.90e-09 | 0.254 | 0.064 | 36 | 45 | all-chemistry |
| EPHA4 | receptor | 181.852 | 1.527 | 8.97e-26 | 0.205 | 0.054 | 44 | 45 | all-chemistry |
| ITGB1 | receptor | 2730.550 | 1.522 | 8.31e-23 | 0.829 | 0.392 | 42 | 45 | all-chemistry |
| CTLA4 | receptor | 248.601 | 1.459 | 1.72e-13 | 0.209 | 0.079 | 41 | 45 | all-chemistry |
| PDCD1 | receptor | 148.444 | 1.430 | 1.06e-08 | 0.170 | 0.055 | 37 | 45 | all-chemistry |
| TFRC | receptor | 112.048 | 1.264 | 1.34e-20 | 0.131 | 0.053 | 43 | 45 | all-chemistry |
| LSR | receptor | 317.553 | 1.107 | 1.25e-08 | 0.278 | 0.107 | 36 | 45 | all-chemistry |
| CD27 | receptor | 1318.419 | 1.037 | 3.45e-12 | 0.678 | 0.523 | 43 | 45 | all-chemistry |

**Top 10 down in malignant**:

| gene | role | baseMean | log2FC | padj | pct_M | pct_B | donors_concordant | donors_eligible | fit |
|---|---|---|---|---|---|---|---|---|---|
| CCL5 | ligand | 2093.716 | -6.700 | 8.73e-81 | 0.018 | 0.184 | 43 | 45 | all-chemistry |
| DPP4 | receptor | 107.282 | -6.300 | 4.06e-81 | 0.007 | 0.193 | 45 | 45 | all-chemistry |
| ITGA4 | receptor | 531.004 | -4.663 | 2.80e-49 | 0.071 | 0.424 | 44 | 45 | all-chemistry |
| LGALS3BP | ligand | 94.222 | -4.320 | 5.14e-38 | 0.016 | 0.120 | 40 | 45 | all-chemistry |
| SLC40A1 | receptor | 114.254 | -2.914 | 1.71e-19 | 0.035 | 0.166 | 39 | 45 | all-chemistry |
| CXCR3 | receptor | 233.546 | -2.795 | 3.29e-15 | 0.076 | 0.123 | 36 | 45 | all-chemistry |
| KLRG1 | receptor | 217.320 | -2.729 | 3.84e-55 | 0.043 | 0.210 | 44 | 45 | all-chemistry |
| CD2 | receptor | 1285.695 | -1.222 | 5.91e-10 | 0.429 | 0.669 | 37 | 45 | all-chemistry |
| FAS | receptor | 148.423 | -1.072 | 1.70e-07 | 0.084 | 0.132 | 33 | 45 | all-chemistry |
| TIMP1 | ligand | 578.864 | -1.071 | 6.47e-08 | 0.256 | 0.356 | 34 | 45 | all-chemistry |

**Gated interactions for those genes** — 163 CD4-gene x cognate x partner rows pass the 10% partner gate (18 CD4-side genes, 13 partner types). The full set for every significant gene is in `interactions_blood.csv` (163 rows).

| cd4_gene | sign | direction | cognate_gene | partner_type | partner_pct | partner_mean_cp10k |
|---|---|---|---|---|---|---|
| CCL5 | lost in M | outgoing | CCR1 | Mono_CD14 | 0.304 | 0.963 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_malignant | 0.254 | 0.905 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_Treg | 0.245 | 0.949 |
| CCL5 | lost in M | outgoing | CXCR3 | Plasma | 0.236 | 0.586 |
| CCL5 | lost in M | outgoing | CCR4 | CD4_Treg | 0.229 | 0.731 |
| CCL5 | lost in M | outgoing | CXCR3 | CD8 | 0.201 | 0.787 |
| CCL5 | lost in M | outgoing | CCR1 | Mono_CD16 | 0.165 | 0.301 |
| CCL5 | lost in M | outgoing | CXCR3 | cDC | 0.139 | 0.186 |
| CCL5 | lost in M | outgoing | CXCR3 | CD4_benign | 0.123 | 0.376 |
| CCR4 | gained in M | incoming | CCL5 | NK_ILC | 0.902 | 40.993 |
| CCR4 | gained in M | incoming | CCL5 | CD8 | 0.710 | 28.975 |
| CCR4 | gained in M | incoming | CCL3 | Mono_CD14 | 0.388 | 12.831 |
| CCR4 | gained in M | incoming | CCL3 | NK_ILC | 0.338 | 3.325 |
| CCR4 | gained in M | incoming | CCL5 | Platelet | 0.309 | 4.954 |
| CCR4 | gained in M | incoming | CCL5 | CD4_benign | 0.184 | 3.195 |
| CCR4 | gained in M | incoming | CCL5 | cDC | 0.143 | 0.476 |
| CCR4 | gained in M | incoming | CCL5 | Plasma | 0.130 | 1.094 |
| CCR4 | gained in M | incoming | CCL2 | Platelet | 0.128 | 1.430 |
| CCR4 | gained in M | incoming | CCL3 | Mono_CD16 | 0.109 | 1.081 |
| CD2 | lost in M | incoming | CD58 | cDC | 0.389 | 0.461 |
| CD2 | lost in M | incoming | CD58 | Mono_CD16 | 0.364 | 0.621 |
| CD2 | lost in M | incoming | CD58 | Mono_CD14 | 0.271 | 0.832 |
| CD2 | lost in M | incoming | CD58 | CD4_Treg | 0.191 | 0.525 |
| CD2 | lost in M | incoming | CD58 | CD8 | 0.191 | 0.527 |
| CD2 | lost in M | incoming | CD58 | NK_ILC | 0.180 | 0.637 |
| CD2 | lost in M | incoming | CD58 | Platelet | 0.170 | 0.420 |
| CD2 | lost in M | incoming | CD58 | Plasma | 0.169 | 0.210 |
| CD2 | lost in M | incoming | CD58 | CD4_malignant | 0.129 | 0.351 |
| CD2 | lost in M | incoming | CD58 | CD4_benign | 0.110 | 0.253 |
| CD27 | gained in M | incoming | CD70 | CD4_malignant | 0.212 | 0.854 |
| CD27 | gained in M | incoming | CD70 | Plasma | 0.130 | 0.257 |
| CD70 | gained in M | outgoing | CD27 | CD4_malignant | 0.678 | 4.423 |
| CD70 | gained in M | outgoing | CD27 | CD4_Treg | 0.617 | 3.227 |
| CD70 | gained in M | outgoing | CD27 | CD4_benign | 0.523 | 1.897 |
| CD70 | gained in M | outgoing | CD27 | CD4_naive | 0.459 | 1.737 |
| CD70 | gained in M | outgoing | CD27 | Plasma | 0.358 | 1.756 |
| CD70 | gained in M | outgoing | CD27 | CD8 | 0.307 | 1.557 |
| CTLA4 | gained in M | incoming | CD86 | cDC | 0.670 | 1.407 |
| CTLA4 | gained in M | incoming | CD86 | Mono_CD16 | 0.643 | 1.648 |
| CTLA4 | gained in M | incoming | CD86 | Mono_CD14 | 0.312 | 1.105 |
| CTLA4 | gained in M | incoming | CD86 | Plasma | 0.149 | 0.204 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_Treg | 0.245 | 0.949 |
| CXCL13 | gained in M | outgoing | CXCR5 | B | 0.239 | 0.780 |
| CXCL13 | gained in M | outgoing | CXCR3 | Plasma | 0.236 | 0.586 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD8 | 0.201 | 0.787 |
| CXCL13 | gained in M | outgoing | CXCR3 | cDC | 0.139 | 0.186 |
| CXCL13 | gained in M | outgoing | CXCR3 | CD4_benign | 0.123 | 0.376 |
| CXCL13 | gained in M | outgoing | CXCR5 | Plasma | 0.122 | 0.147 |
| CXCR3 | lost in M | incoming | CCL5 | NK_ILC | 0.902 | 40.993 |
| CXCR3 | lost in M | incoming | CCL5 | CD8 | 0.710 | 28.975 |
| CXCR3 | lost in M | incoming | CCL5 | Platelet | 0.309 | 4.954 |
| CXCR3 | lost in M | incoming | CXCL12 | Platelet | 0.223 | 3.024 |
| CXCR3 | lost in M | incoming | PF4 | Platelet | 0.184 | 2.889 |
| CXCR3 | lost in M | incoming | CCL5 | CD4_benign | 0.184 | 3.195 |
| CXCR3 | lost in M | incoming | CCL5 | cDC | 0.143 | 0.476 |
| CXCR3 | lost in M | incoming | CXCL13 | CD4_malignant | 0.137 | 0.857 |
| CXCR3 | lost in M | incoming | CCL5 | Plasma | 0.130 | 1.094 |
| EPHA4 | gained in M | incoming | EFNB1 | cDC | 0.254 | 0.280 |
| EPHA4 | gained in M | incoming | EFNA5 | Platelet | 0.120 | 0.820 |
| EPHA4 | gained in M | incoming | EFNB2 | Platelet | 0.100 | 0.345 |
| FAS | lost in M | incoming | TNFSF13 | Mono_CD16 | 0.558 | 1.315 |
| FAS | lost in M | incoming | TNFSF13 | cDC | 0.439 | 0.712 |
| FAS | lost in M | incoming | TNFSF13 | Mono_CD14 | 0.280 | 0.788 |
| FAS | lost in M | incoming | FASLG | NK_ILC | 0.157 | 0.574 |
| FGFR1 | gained in M | incoming | NCAM1 | NK_ILC | 0.217 | 0.881 |
| FGFR1 | gained in M | incoming | FGF7 | Platelet | 0.147 | 1.142 |
| IGFL2 | gained in M | outgoing | IGFLR1 | cDC | 0.607 | 0.926 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Plasma | 0.386 | 0.651 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Mono_CD16 | 0.333 | 0.553 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Mono_CD14 | 0.289 | 0.963 |
| IGFL2 | gained in M | outgoing | IGFLR1 | B | 0.227 | 0.750 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_Treg | 0.220 | 0.642 |
| IGFL2 | gained in M | outgoing | IGFLR1 | NK_ILC | 0.196 | 0.771 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_naive | 0.176 | 0.519 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD8 | 0.161 | 0.426 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_benign | 0.155 | 0.352 |
| IGFL2 | gained in M | outgoing | IGFLR1 | CD4_malignant | 0.138 | 0.379 |
| IGFL2 | gained in M | outgoing | IGFLR1 | Platelet | 0.124 | 0.365 |
| ITGA4 | lost in M | incoming | CD14 | Mono_CD14 | 0.769 | 11.798 |
| ITGA4 | lost in M | incoming | VCAN | Mono_CD14 | 0.768 | 12.909 |

## 7. Skin vs blood

Significant in skin only: **20**; blood only: **20**; both: **7**.

- shared: CCL5, CD70, CXCL13, FAS, IGFL2, KIR3DL2, PTHLH
- skin only: AREG, CCR6, CD151, CD28, CD4, CD40LG, CD5, CD55, CD9, CXCR4, FLT3LG, ICOS, IL10RA, KCNA3, LTB, PDGFA, PTH2R, SELL, SORL1, TNFSF8
- blood only: CCR4, CD2, CD27, CTLA4, CXCR3, DPP4, EPHA4, FGFR1, IGFBP4, ITGA4, ITGB1, KLRG1, LGALS3BP, LSR, PDCD1, SLC40A1, TFRC, TIMP1, TNFSF11, TNFSF9

Across the **271 LR genes tested in both tissues**, the shrunken log2FC correlate at Pearson r = **0.678** (Spearman 0.540), and 184 of 271 (68%) agree in sign.

**Significance in one tissue only is not evidence of tissue specificity.** The two compartments differ in eligible donor count, in cell numbers per arm, and in the composition of the benign arm, so they differ in power; a gene can miss the threshold in one tissue while its estimate is the same size and sign in both. Only the effect-size comparison above speaks to tissue specificity, and it does so weakly.

## 8. Prior axes

Reported regardless of significance, both tissues. `padj_family` says whether the gene is inside the BH family (the filtered LR resource); five of the named axes are not, and their cognate pairs are resolved against the unfiltered consensus instead. **TOX is the positive control for the malignant contrast**, not an LR gene.

| gene | tissue | log2FC | padj | padj_family | pct_M | pct_B | donors_concordant | donors_eligible | fit | cognate_genes | gated_partners |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CCR4 | blood | 1.566 | 9.90e-09 | LR genes | 0.254 | 0.064 | 36 | 45 | all-chemistry | CCL17; CCL22 |  |
| CCR4 | skin | -0.476 | 0.011 | LR genes | 0.266 | 0.248 | 28 | 56 | all-chemistry | CCL17; CCL22 | CCL17:Myeloid (23%); CCL22:Myeloid (29%) |
| CD27 | blood | 1.037 | 3.45e-12 | LR genes | 0.678 | 0.523 | 43 | 45 | all-chemistry | CD70 | CD70:CD4_malignant (21%); CD70:Plasma (13%) |
| CD27 | skin | -0.914 | 1.71e-04 | LR genes | 0.370 | 0.228 | 30 | 56 | all-chemistry | CD70 | CD70:B (13%); CD70:CD4_Treg (12%); CD70:CD4_malignant (22%); CD70:Plasma (11%); CD70:Prolif (26%) |
| CD28 | blood | 0.772 | 3.85e-16 | LR genes | 0.401 | 0.270 | 40 | 45 | all-chemistry | CD80; CD86 | CD86:Mono_CD14 (31%); CD86:Mono_CD16 (64%); CD86:Plasma (15%); CD86:cDC (67%) |
| CD28 | skin | -1.044 | 9.07e-06 | LR genes | 0.309 | 0.274 | 36 | 56 | all-chemistry | CD80; CD86 | CD80:Myeloid (19%); CD86:Myeloid (46%) |
| CD70 | blood | 2.960 | 4.22e-31 | LR genes | 0.212 | 0.023 | 41 | 45 | all-chemistry | CD27 | CD27:CD4_Treg (62%); CD27:CD4_benign (52%); CD27:CD4_malignant (68%); CD27:CD4_naive (46%); CD27:CD8 (31%); CD27:Plasma (36%) |
| CD70 | skin | 1.961 | 3.92e-23 | LR genes | 0.223 | 0.030 | 51 | 56 | all-chemistry | CD27 | CD27:B (16%); CD27:CD4_Treg (55%); CD27:CD4_benign (23%); CD27:CD4_malignant (37%); CD27:CD8 (40%); CD27:NK_ILC (11%); CD27:Plasma (45%); CD27:Prolif (51%); CD27:gdT (25%) |
| CSF2 | blood |  |  | none (outside the filtered resource) |  |  | 0 | 0 |  | CSF1R; CSF2RA_CSF2RB; CSF3R; IL3RA; ITGA9; ITGB1; SDC2 | CSF1R:Mono_CD14 (34%); CSF1R:Mono_CD16 (88%); CSF1R:cDC (40%); CSF2RA_CSF2RB:Mono_CD14 (21%); CSF2RA_CSF2RB:Mono_CD16 (22%); CSF2RA_CSF2RB:cDC (21%); CSF3R:Mono_CD14 (67%); CSF3R:Mono_CD16 (32%); CSF3R:cDC (56%); IL3RA:Mono_CD14 (11%); IL3RA:Mono_CD16 (20%); IL3RA:Plasma (21%) |
| CSF2 | skin | 0.341 | 0.030 | LR genes | 0.095 | 0.024 | 34 | 56 | all-chemistry | CSF1R; CSF2RA_CSF2RB; CSF3R; IL3RA; ITGA9; ITGB1; SDC2 | CSF1R:Myeloid (29%); CSF2RA_CSF2RB:Myeloid (26%); CSF2RA_CSF2RB:Plasma (11%); CSF3R:Myeloid (10%); IL3RA:Myeloid (19%); IL3RA:Plasma (27%); IL3RA:Vascular (45%); ITGA9:Vascular (17%); ITGB1:B (12%); ITGB1:CD4_Treg (42%); ITGB1:CD4_benign (35%); ITGB1:CD4_malignant (42%) |
| CTLA4 | blood | 1.459 | 1.72e-13 | LR genes | 0.209 | 0.079 | 41 | 45 | all-chemistry | CD80; CD86 | CD86:Mono_CD14 (31%); CD86:Mono_CD16 (64%); CD86:Plasma (15%); CD86:cDC (67%) |
| CTLA4 | skin | -0.383 | 0.040 | LR genes | 0.303 | 0.159 | 33 | 56 | all-chemistry | CD80; CD86 | CD80:Myeloid (19%); CD86:Myeloid (46%) |
| CXCR4 | blood | -0.773 | 1.08e-06 | LR genes | 0.535 | 0.590 | 30 | 45 | all-chemistry | CXCL12 | CXCL12:Platelet (22%) |
| CXCR4 | skin | -1.769 | 2.00e-42 | LR genes | 0.545 | 0.722 | 55 | 56 | all-chemistry | CXCL12 | CXCL12:Fibroblast (61%); CXCL12:Vascular (24%) |
| IL22 | blood |  |  | none (outside the filtered resource) |  |  | 0 | 0 |  | IL22RA1 |  |
| IL22 | skin | -2.145 | 2.86e-06 | LR genes | 0.036 | 0.008 | 31 | 56 | all-chemistry | IL22RA1 |  |
| JAML | blood | -0.920 |  | none (outside the filtered resource) | 0.114 | 0.147 | 29 | 45 | all-chemistry | CXADR |  |
| JAML | skin | -0.359 |  | none (outside the filtered resource) | 0.285 | 0.203 | 32 | 56 | all-chemistry | CXADR | CXADR:Keratinocyte (43%) |
| KIR3DL2 | blood | 4.660 | 1.60e-41 | LR genes | 0.369 | 0.010 | 41 | 45 | all-chemistry | HLA-A | HLA-A:B (96%); HLA-A:CD4_Treg (99%); HLA-A:CD4_benign (99%); HLA-A:CD4_malignant (99%); HLA-A:CD4_naive (99%); HLA-A:CD8 (99%); HLA-A:Erythroid (50%); HLA-A:Mono_CD14 (96%); HLA-A:Mono_CD16 (99%); HLA-A:NK_ILC (100%); HLA-A:Plasma (92%); HLA-A:Platelet (80%) |
| KIR3DL2 | skin | 1.008 | 0.004 | LR genes | 0.175 | 0.009 | 22 | 56 | non-Flex | HLA-A | HLA-A:B (34%); HLA-A:CD4_Treg (44%); HLA-A:CD4_benign (24%); HLA-A:CD4_malignant (52%); HLA-A:CD8 (35%); HLA-A:Fibroblast (66%); HLA-A:Keratinocyte (59%); HLA-A:Mast (43%); HLA-A:Melanocyte (40%); HLA-A:Myeloid (44%); HLA-A:NK_ILC (51%); HLA-A:Plasma (29%) |
| SPN | blood | -0.092 |  | none (outside the filtered resource) | 0.299 | 0.310 | 19 | 45 | all-chemistry | SIGLEC1 |  |
| SPN | skin | 0.367 |  | none (outside the filtered resource) | 0.344 | 0.131 | 45 | 56 | all-chemistry | SIGLEC1 | SIGLEC1:Myeloid (10%) |
| TOX | blood | 2.928 |  | none (outside the filtered resource) | 0.449 | 0.063 | 44 | 45 | all-chemistry |  |  |
| TOX | skin | 1.151 |  | none (outside the filtered resource) | 0.490 | 0.092 | 49 | 56 | all-chemistry |  |  |

## 9. Limitations

- **No proximity information.** Dissociated scRNA-seq says nothing about whether two cells were ever in contact. Every interaction reported here is a statement about co-expression within a tissue, not about an engagement.
- **Partner assignments are not corrected for ambient RNA and are suggestive only.** No ambient correction was performed anywhere in this analysis, by design. A partner type that sits next to a highly expressing neighbour will carry that neighbour's transcripts, and the gate is a detection fraction, which ambient contamination inflates directly.
- **Benign arm composition.** The benign arm is whatever the malignancy call left behind. In skin that is conventional CD4 minus the `CD4_Treg` level; in blood it is conventional CD4 after the Treg-like and naive-like clusters were removed by a marker rule, which is a cruder instrument than a curated label. Residual activation-state differences between the arms are not separable from malignancy here.
- **Contact-dependent interactions in blood are weakly interpretable.** Circulating cells are not in stable contact. A gained CD28–CD86 or CTLA4–CD80 edge in blood is a statement about the transcript, not about a synapse.
- **Number of eligible blood donors.** Blood has fewer eligible donors than skin, and the geskin2026 exclusion removed the largest blood cohort outright. Power is correspondingly lower, which is the main reason a skin-only hit is not a tissue-specificity claim.
- **The malignancy call is a cluster call.** `mal_cluster` assigns a whole per-sample cluster, so a benign cell inside a malignant cluster is labelled malignant and vice versa. `helpers/malignancy.py` also notes that healthy-control samples are called benign by construction, so a 0% HC rate is an input, not a result.

## 10. Decisions and deviations; upstream errors noticed

### Decisions and deviations from the brief

- **The filtered LR resource was rebuilt, not loaded — it does not exist on disk.** The brief names it as an artefact of a rebuilt LIANA+ pipeline. A scoped search of `data/`, `tables/`, `docs/` and `helpers/` found no such file; `helpers/ccc_panel.py:12` carries the specification as a run note and nothing implements it. It was reproduced exactly: LIANA `consensus` intersected with support >= 6 of the 15 human resources gives 1,422 interactions over 1,043 genes, and the script asserts both numbers.
- **No blacklist was applied.** No blacklist file exists in this repo, and `docs/CCC_PIPELINE_PORTABLE_SPEC.md:83` states that `LR_CAVEATS` is *commentary only, never a blacklist*, because the v1 version silently deleted the IL4/IL13 pairs the analysis existed to report (`helpers/ccc_utils.py:1157`). Applying one would have contradicted the repo's own standing instruction, so the caveat list is not used as a filter.
- **Arms come from `mal_cluster`, not from the inferCNV/TCR `combined_malignant` flag.** The brief asks for 'the final malignancy calls (inferCNV/TCR pipeline)'. That combined flag was retired: `helpers/malignancy.py` states the four calls are never OR-ed, because the CNV call derives from smoothed regional expression — partly the same signal a downstream expression analysis then attributes. `mal_cluster` has been the headline since 2026-10-22 and is anchored per sample on a TCR sequence fact.
- **The unit is the curated person key.** `clin_person_key` falling back to `patient_key` and then `donor`, so longitudinal and multi-site samples of one patient are summed and the two patients recorded under two donor keys are merged.
- **geskin2026 was dropped from blood in full**, on the user's instruction. All 260,455 of its cells are ex vivo 2-day cultures (213,181 drug-treated, 47,274 vehicle), which are not in-vivo biology.
- **The benign arm is the re-annotated lineage, not `cell_type_final`.** `CD4_conv` + `CD4_naive` in skin, `CD4_conv` in blood, from `nonmal_T_labels_v1` via `helpers/lineage.py`. The v1 blood Treg/naive cluster clean-up is gone — the re-annotation subsumes it, in both tissues. `arm_composition.csv` has the full old-label x new-label crosstab.
- **Counts were taken from `joint_annotated.h5ad`, not the compartment objects**, so that blood is analysed over the full 42,347-gene space rather than a 10,000-gene HVG subset.
- **Five prior axes fall outside the filtered resource** (JAML–CXADR, SPN–SIGLEC1, IL22–IL22RA1, CSF2's receptor, and TOX, which is not an LR gene). They are reported from the unfiltered consensus and flagged, since the brief asks for them regardless of significance.
- **DESeq2's own independent-filtering padj is discarded.** Its family is the whole transcriptome; the brief asks for BH over the LR resource only, so BH is recomputed from the Wald p-values over that family, per tissue per fit.
- **The 'same person under two donor keys' cohort rule is applied as a merge, not a drop**, because patient-level summation is what the brief asks for.
- **Carlito and Calibri are not installed on these compute nodes.** They remain first in the font stack, so the figures render in them anywhere the fonts exist; here they fall back to DejaVu Sans. Everything else in the figure specification — PNG and PDF at 400 dpi, `pdf.fonttype=42`, white background, the single teal accent #0F766E, no decorative elements — is as requested.

### Upstream data errors and inconsistencies noticed

- **The brief's description of geskin2026 does not match the atlas.** It says 5 patients and 28 units. On disk geskin2026 is blood-only, 260,455 cells, 11 `patient_key` values, 65 `donor` keys that are lane × HTO wells, and 11 libraries. `docs/DATA_PROVENANCE.md:61` records that the HTO-to-patient map was never deposited, so the 65 donor keys are pseudo-donors, not people.
- **`blood_T_annotated.h5ad` is a 10,000-gene HVG subset while `skin_T_annotated.h5ad` carries all 42,347.** This asymmetry is not flagged in either object. Using the compartment objects naively would have tested 590 of 1,043 LR genes in blood against 1,028 in skin, and would have silently dropped CXADR and IL22RA1 from the prior-axes table.
- **The blood NK/CD8 labels are mixtures.** The blood object holds 606 NK against 112,609 CD8. `README.md` states that nb31 called CD8 on `CD8A`, which NK cells express as a homodimer, and that `blood_T_celltype_corrections_v1.csv` corrects only the 12 clusters the nb32b review examined while roughly 100 further CD8-labelled clusters carry the same signature. The blood CD8 and NK partner rows should be read as mixtures.
- **`atlas_obs_full_v2.parquet` carries stale columns.** Its `has_tcr` is wrong for li2024 (`helpers/malignancy.py` documents 0.072 there against 0.476 in the V(D)J fold-in cache), and its `is_malignant` is an older atlas-level call that disagrees with every current one. Neither is used here.
- **`_v2`/`_v3` filename suffixes are not recency markers** in this tree — `CLAUDE.md` records that `skin_T_annotated_v2.h5ad` was older than the unsuffixed file. Every input above was pinned by reading the module that owns it, not by its suffix.

## 11. File manifest

| file | contents |
|---|---|
| ccc_malignant_vs_benign_cd4_report.md | this report |
| de_lr_genes_skin.csv | every tested LR gene in skin, with the top-hit columns |
| de_lr_genes_blood.csv | every tested LR gene in blood |
| de_all_genes_skin.csv | the full-transcriptome fit for skin (no BH outside the LR family) |
| de_all_genes_blood.csv | the full-transcriptome fit for blood |
| interactions_skin.csv | CD4 gene × cognate gene × partner type passing the gate, skin |
| interactions_blood.csv | the same for blood |
| cell_counts_donor_arm.csv | cells per donor × tissue × arm, with eligibility |
| prior_axes.csv | the pre-specified axes plus the TOX control, both tissues |
| partner_types.csv | every candidate partner type with its pooled cell count and the keep flag |
| partners.csv | the partner types that passed the cell-count floor, with their indices |
| cohort_excluded.csv | every unit removed by a cohort rule, with the reason |
| arm_composition.csv | v1 label × v2 label × malignancy × arm, per tissue — exactly what the re-annotation moved |
| annotation_shift.csv | per gene, the v1 and v2 effect size side by side with a stable / moved verdict |
| units.csv | the eligible donor × arm pseudobulk index |
| lr_resource_1422.csv | the rebuilt filtered resource (1,422 interactions) |
| lr_resource_consensus_full.csv | unfiltered LIANA consensus with per-pair support |
| lr_genes.json | the 1,043 unique genes of the filtered resource |
| _expected_unit_cells.csv | cells per pseudobulk, written by `prep` and asserted by `pass` |
| pseudobulk_skin.parquet | donor × arm summed raw counts, skin (full gene space) |
| pseudobulk_blood.parquet | donor × arm summed raw counts, blood (full gene space) |
| pass_accumulators.npz | the streamed count sums and detection counts |
| row_labels.npz | cell → (donor × arm) and cell → partner-type label vectors |

_Figures are opt-in: the `fit` stage writes them only when run with `--figures`._

`for_analysis/` holds the text-only subset of the above (no figures, no `.npz`, no parquet) with its own `INDEX.md`, for handing the result to another model or reader. It is rewritten by every `fit` run, and by `run_ccc52_mvb.sh bundle` on its own.

Produced by `jobs/run_ccc52_mvb.py` (stages `lr`, `prep`, `pass`, `fit`), submitted with `jobs/run_ccc52_mvb.sh <stage>`. Nothing under `data/` was written.